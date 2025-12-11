import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import copy
import numpy as np
from tqdm import tqdm

from transformers.file_utils import is_tf_available, is_torch_available
from transformers.tokenization_bert import whitespace_tokenize
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import logging
from transformers.data.processors.utils import DataProcessor
import collections

# Store the tokenizers which insert 2 separators tokens
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart"}

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

logger = logging.get_logger(__name__)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def squad_convert_example_to_features(
        example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training
):

    # 初始化特征列表，每个元素对应一个处理后的文本片段（span）
    features = []
    # 训练模式且答案存在时的处理
    if is_training and not example.is_impossible:
        # 获取答案的原始起止位置
        start_position = example.start_position
        end_position = example.end_position

        # 从文档中提取实际答案文本并清理空格
        actual_text = " ".join(example.doc_tokens[start_position: (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        # 验证实际文本是否包含答案，若不存在则跳过该样本
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    # 初始化分词后的索引映射和文档tokens
    tok_to_orig_index = []# 每个subtoken对应的原始token索引
    orig_to_tok_index = []# 每个原始token的第一个subtoken索引
    all_doc_tokens = []# 所有subtoken组成的文档
    for (i, token) in enumerate(example.doc_tokens):# 记录当前原始token的起始subtoken位置
        orig_to_tok_index.append(len(all_doc_tokens))# 分词为subtoken（如WordPiece）
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)# 当前subtoken属于第i个原始token
            all_doc_tokens.append(sub_token) # 将subtoken加入列表

    # 训练模式下调整答案的subtoken起止位置
    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]# 答案起始对应的subtoken位置
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        
        # 优化答案的subtoken范围，确保准确匹配答案文本
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []# 存储所有处理后的文本片段

    # 将问题文本编码为ID，并截断到最大允许长度
    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    # 根据tokenizer类型确定添加的特殊token数量（如[SEP]）
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    # all_doc_tokens=all_doc_tokens[:tok_start_position] + ["[unused100]"] + all_doc_tokens[tok_start_position:tok_end_position+1] + ["[unused101]"] + all_doc_tokens[tok_end_position+1:]
    # 使用滑动窗口将长文档分割为多个片段
    span_doc_tokens = all_doc_tokens# 初始为完整文档
    while len(spans) * doc_stride < len(all_doc_tokens):# 直到覆盖整个文档

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query # 问题作为第一个句子
            pairs = span_doc_tokens # 文档片段作为第二个句子
            truncation = TruncationStrategy.ONLY_SECOND.value# 仅截断文档
        else:
            texts = span_doc_tokens # 左padding时文档在前
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value

         # 编码组合后的文本，处理溢出
        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            texts,
            pairs,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=True, # 返回溢出部分供后续处理
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )

        # 计算当前片段的有效文档长度（扣除问题和特殊token占用的位置）
        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        # 去除padding部分，获取实际token IDs
        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                        len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(
                    tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)# 转换为token字符串

        # 构建token到原始文档的映射
        token_to_orig_map = {}
        for i in range(paragraph_len):
             # 根据padding方向确定当前token在输入中的位置
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        # 将当前片段的信息存入字典
        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {} # 后续填充每个token是否处于最大上下文
        encoded_dict["start"] = len(spans) * doc_stride# 当前片段在文档中的起始位置
        encoded_dict["length"] = paragraph_len # 实际文档长度

        spans.append(encoded_dict)# 将当前片段加入列表

        # 如果没有溢出token，结束循环；否则处理剩余的token
        if "overflowing_tokens" not in encoded_dict or (
                "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    # 标记每个token是否处于最大上下文窗口（避免滑动窗口中的重复）
    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    # 将每个span转换为模型输入特征
    for span in spans:
        # Identify the position of the CLS token
        # 定位CLS token的位置
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        # 创建p_mask，标记不允许作为答案的token（1为屏蔽，0为允许）
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            # 右padding时，问题部分和特殊token之后的位置允许作为答案
            p_mask[len(truncated_query) + sequence_added_tokens:] = 0
        else:
            # 左padding时，中间部分允许作为答案
            p_mask[-len(span["tokens"]): -(len(truncated_query) + sequence_added_tokens)] = 0

        # 将padding和特殊token的位置设为屏蔽
        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0# CLS位置可用于标记无法回答的问题

        # 处理训练时的标签
        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            # 检查答案是否在当前span中
            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True# 答案不在当前span，标记为不可能

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                # 计算答案在span内的相对位置（考虑padding方向）
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        # 将当前span的特征添加到结果列表
        features.append(
            SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,
                # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
            )
        )
    return features


def squad_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def squad_convert_examples_to_features(
        examples,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        padding_strategy="max_length",
        return_dataset=False,
        threads=1,
        tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        padding_strategy: Default to "max_length". Which padding strategy to use
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    #多线程并行处理​
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
                disable=not tqdm_enabled,
            )
        )
    # 特征后处理​
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
            features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        if not is_training:
            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
            all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask
            )
            return features, dataset
        else:
            ##初始化数据结构
            new_features=[]
            example_index_to_features = collections.defaultdict(list)#建立字典
            for feature in features:
                example_index_to_features[feature.example_index].append(feature)

            all_input_ids = []
            all_attention_masks = []
            all_token_type_ids = []
            all_start_positions = []
            all_end_positions = []
            all_qve_input_ids = []
            all_qve_attention_masks = []
            all_qve_token_type_ids = []
            special_tokens_ids = tokenizer.additional_special_tokens_ids

            #特征处理
            for ex, efs in example_index_to_features.items():
                tmp_feature = None
                for ef in efs:#找出最合适的特征
                    if ef.start_position > 0 and ef.end_position > 0:
                        if tmp_feature:
                            if ef.start_position < tmp_feature.start_position:
                                tmp_feature = ef
                        else:
                            tmp_feature = ef

                if not tmp_feature:
                    tmp_feature = efs[0]

                #计算问题和段落特征以便进行重新编码
                question_len = tmp_feature.input_ids.index(tokenizer.sep_token_id) + 1
                new_para_ids = tmp_feature.tokens[question_len:-1]

                new_question_ids = tmp_feature.input_ids[1:question_len - 1]

                if tmp_feature.start_position == tmp_feature.end_position == 0:
                    new_question_ids = new_question_ids + [special_tokens_ids[0]] + [special_tokens_ids[1]]
                else:
                    new_question_ids = new_question_ids + [special_tokens_ids[0]] + \
                                       tmp_feature.input_ids[tmp_feature.start_position:tmp_feature.end_position + 1]

                #创建QVE模型输入
                encoded_dict = tokenizer.encode_plus(
                    new_question_ids,
                    new_para_ids,
                    truncation=True,
                    padding=padding_strategy,
                    max_length=max_seq_length,
                    return_token_type_ids=True,
                )

               
                qve_input_id = encoded_dict.data['input_ids']
                qve_attention_mask = encoded_dict.data['attention_mask']
                qve_token_type_id = encoded_dict.data['token_type_ids']
               

                if tmp_feature.start_position==0 and tmp_feature.end_position==0:
                    continue

                 #创建PyTorch张量
                all_qve_input_ids.append(qve_input_id)
                all_qve_attention_masks.append(qve_attention_mask)
                all_qve_token_type_ids.append(qve_token_type_id)

                all_input_ids.append(tmp_feature.input_ids)
                all_attention_masks.append(tmp_feature.attention_mask)
                all_token_type_ids.append(tmp_feature.token_type_ids)
                all_start_positions.append(tmp_feature.start_position)
                all_end_positions.append(tmp_feature.end_position)
                new_features.append(tmp_feature)


            all_qve_input_ids = torch.tensor(all_qve_input_ids, dtype=torch.long)
            all_qve_attention_masks = torch.tensor(all_qve_attention_masks, dtype=torch.long)
            all_qve_token_type_ids = torch.tensor(all_qve_token_type_ids, dtype=torch.long)

            all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
            all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
            all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
            all_start_positions = torch.tensor(all_start_positions, dtype=torch.long)
            all_end_positions = torch.tensor(all_end_positions, dtype=torch.long)

            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_feature_index,
                all_start_positions,
                all_end_positions,
                all_qve_input_ids,
                all_qve_attention_masks,
                all_qve_token_type_ids,
            )
            features = new_features
            return features, dataset
    elif return_dataset == "tf":
        if not is_tf_available():
            raise RuntimeError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for i, ex in enumerate(features):
                if ex.token_type_ids is None:
                    yield (
                        {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "feature_index": i,
                            "qas_id": ex.qas_id,
                        },
                        {
                            "start_positions": ex.start_position,
                            "end_positions": ex.end_position,
                            "cls_index": ex.cls_index,
                            "p_mask": ex.p_mask,
                            "is_impossible": ex.is_impossible,
                        },
                    )
                else:
                    yield (
                        {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                            "feature_index": i,
                            "qas_id": ex.qas_id,
                        },
                        {
                            "start_positions": ex.start_position,
                            "end_positions": ex.end_position,
                            "cls_index": ex.cls_index,
                            "p_mask": ex.p_mask,
                            "is_impossible": ex.is_impossible,
                        },
                    )

        # Why have we split the batch into a tuple? PyTorch just has a list of tensors.
        if "token_type_ids" in tokenizer.model_input_names:
            train_types = (
                {
                    "input_ids": tf.int32,
                    "attention_mask": tf.int32,
                    "token_type_ids": tf.int32,
                    "feature_index": tf.int64,
                    "qas_id": tf.string,
                },
                {
                    "start_positions": tf.int64,
                    "end_positions": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            )

            train_shapes = (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                    "feature_index": tf.TensorShape([]),
                    "qas_id": tf.TensorShape([]),
                },
                {
                    "start_positions": tf.TensorShape([]),
                    "end_positions": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            )
        else:
            train_types = (
                {"input_ids": tf.int32, "attention_mask": tf.int32, "feature_index": tf.int64, "qas_id": tf.string},
                {
                    "start_positions": tf.int64,
                    "end_positions": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            )

            train_shapes = (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "feature_index": tf.TensorShape([]),
                    "qas_id": tf.TensorShape([]),
                },
                {
                    "start_positions": tf.TensorShape([]),
                    "end_positions": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            )

        return tf.data.Dataset.from_generator(gen, train_types, train_shapes)
    else:
        return features

def squad_convert_3p_example_to_features(
        example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training, input_mode
):

    # 初始化特征列表，每个元素对应一个处理后的文本片段（span）
    features = []
    # 训练模式且答案存在时的处理
    if is_training and not example.is_impossible:
        # 获取答案的原始起止位置
        start_position = example.start_position
        end_position = example.end_position

        # 从文档中提取实际答案文本并清理空格
        actual_text = " ".join(example.doc_tokens[start_position: (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        # 验证实际文本是否包含答案，若不存在则跳过该样本
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    # 初始化分词后的索引映射和文档tokens
    tok_to_orig_index = []# 每个subtoken对应的原始token索引
    orig_to_tok_index = []# 每个原始token的第一个subtoken索引
    all_doc_tokens = []# 所有subtoken组成的文档
    for (i, token) in enumerate(example.doc_tokens):# 记录当前原始token的起始subtoken位置
        orig_to_tok_index.append(len(all_doc_tokens))# 分词为subtoken（如WordPiece）
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)# 当前subtoken属于第i个原始token
            all_doc_tokens.append(sub_token) # 将subtoken加入列表

    # 训练模式下调整答案的subtoken起止位置
    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]# 答案起始对应的subtoken位置
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        
        # 优化答案的subtoken范围，确保准确匹配答案文本
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []# 存储所有处理后的文本片段

    # 将问题文本编码为ID，并截断到最大允许长度
    truncated_query = tokenizer.encode(
        example.question_text ,add_special_tokens=False, truncation=True, max_length=max_query_length
    )
    truncated_answer = tokenizer.encode(
        example.answer_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )
    q_start = 1  # [CLS]之后
    q_end = q_start + len(truncated_query)
    a_end = q_end + len(truncated_answer)

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    # 根据tokenizer类型确定添加的特殊token数量（如[SEP]）
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    # all_doc_tokens=all_doc_tokens[:tok_start_position] + ["[unused100]"] + all_doc_tokens[tok_start_position:tok_end_position+1] + ["[unused101]"] + all_doc_tokens[tok_end_position+1:]
    # 使用滑动窗口将长文档分割为多个片段
    span_doc_tokens = all_doc_tokens# 初始为完整文档
    while len(spans) * doc_stride < len(all_doc_tokens):# 直到覆盖整个文档

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query +  truncated_answer# 问题作为第一个句子
            pairs = span_doc_tokens # 文档片段作为第二个句子
            truncation = TruncationStrategy.ONLY_SECOND.value# 仅截断文档
        else:
            texts = span_doc_tokens # 左padding时文档在前
            pairs = truncated_query +  truncated_answer
            truncation = TruncationStrategy.ONLY_FIRST.value

         # 编码组合后的文本，处理溢出
        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            texts,
            pairs,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=True, # 返回溢出部分供后续处理
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )

        # 计算当前片段的有效文档长度（扣除问题和特殊token占用的位置）
        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        # 去除padding部分，获取实际token IDs
        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                        len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(
                    tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)# 转换为token字符串

        # 构建token到原始文档的映射
        token_to_orig_map = {}
        for i in range(paragraph_len):
             # 根据padding方向确定当前token在输入中的位置
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        # 将当前片段的信息存入字典
        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {} # 后续填充每个token是否处于最大上下文
        encoded_dict["start"] = len(spans) * doc_stride# 当前片段在文档中的起始位置
        encoded_dict["length"] = paragraph_len # 实际文档长度

        encoded_dict["padding_side"] = tokenizer.padding_side
        if tokenizer.padding_side == "right":
            # 结构：[CLS] 查询 [SEP] 文档 [SEP]
            # [CLS]位置为0，第二个token开始是查询部分
            encoded_dict["q_start"] = 1
            encoded_dict["q_end"] = 1 + len(truncated_query) - 1
            # 答案紧接着查询
            encoded_dict["a_start"] = encoded_dict["q_end"] + 1
            encoded_dict["a_end"] = encoded_dict["a_start"] + len(truncated_answer) - 1
            # 文档在第一个[SEP]之后
            sep1_index = len(truncated_query) + len(truncated_answer) + 1  # [SEP]位置
            encoded_dict["text_start"] = sep1_index + 1
            # 文档结束位置需要动态查找第二个[SEP]
            try:
                # 尝试在输入ID中查找第二个[SEP]
                sep2_index = non_padded_ids.index(tokenizer.sep_token_id, sep1_index + 1)
                encoded_dict["text_end"] = sep2_index - 1
            except ValueError:
                # 如果找不到第二个SEP，则使用段落长度计算结束位置
                encoded_dict["text_end"] = encoded_dict["text_start"] + paragraph_len - 1
        
        else:
            # 结构：文档 [SEP] 查询 [SEP]（左侧可能填充）
            # [CLS]位置为0，文档从1开始
            encoded_dict["text_start"] = 1
            # 文档结束位置在第一个[SEP]之前
            sep1_index = 1 + paragraph_len  # 第一个[SEP]位置
            encoded_dict["text_end"] = sep1_index - 1
            # 查询在第一个[SEP]之后
            encoded_dict["q_start"] = sep1_index + 1
            encoded_dict["q_end"] = encoded_dict["q_start"] + len(truncated_query) - 1
            # 答案紧接着查询
            encoded_dict["a_start"] = encoded_dict["q_end"] + 1
            # 答案结束位置需要动态查找第二个[SEP]
            try:
                # 尝试在输入ID中查找第二个[SEP]
                sep2_index = non_padded_ids.index(tokenizer.sep_token_id, sep1_index + 1)
                encoded_dict["a_end"] = sep2_index - 1
            except ValueError:
                # 如果找不到第二个SEP，则使用基本计算
                encoded_dict["a_end"] = encoded_dict["a_start"] + len(truncated_answer) - 1


        spans.append(encoded_dict)# 将当前片段加入列表

        # 如果没有溢出token，结束循环；否则处理剩余的token
        if "overflowing_tokens" not in encoded_dict or (
                "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    # 标记每个token是否处于最大上下文窗口（避免滑动窗口中的重复）
    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    # 将每个span转换为模型输入特征
    for span in spans:
        # Identify the position of the CLS token
        # 定位CLS token的位置
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        # 创建p_mask，标记不允许作为答案的token（1为屏蔽，0为允许）
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            # 右padding时，问题部分和特殊token之后的位置允许作为答案
            p_mask[len(truncated_query) + sequence_added_tokens:] = 0
        else:
            # 左padding时，中间部分允许作为答案
            p_mask[-len(span["tokens"]): -(len(truncated_query) + sequence_added_tokens)] = 0

        # 将padding和特殊token的位置设为屏蔽
        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0# CLS位置可用于标记无法回答的问题


        # new_mask = span["attention_mask"].copy()
        # if (input_mode == "qa"):
        #     st_pos = span["text_start"]
        #     end_pos = span["text_end"]
        # if (input_mode == "qc"):
        #     st_pos = span["a_start"]
        #     end_pos = span["a_end"]
        # if (input_mode == "ca"):
        #     st_pos = span["q_start"]
        #     end_pos = span["q_end"]
        # slice_length = end_pos - st_pos
        # zeros_list = [0] * slice_length 
        # new_mask[st_pos:end_pos] = zeros_list  
        # #logger.info("new_mask修改后: %s", new_mask) 
        # span["attention_mask"] = new_mask

        # 处理训练时的标签
        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            # 检查答案是否在当前span中
            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True# 答案不在当前span，标记为不可能

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                # 计算答案在span内的相对位置（考虑padding方向）
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        # 将当前span的特征添加到结果列表
        features.append(
            SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,
                # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
            )
        )
    return features


def squad_convert_3p_examples_to_features(
        examples,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        is_training,
        padding_strategy="max_length",
        return_dataset=False,
        threads=1,
        tqdm_enabled=True,
):
    
    # Defining helper methods
    #多线程并行处理​
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
                disable=not tqdm_enabled,
            )
        )
    # 特征后处理​
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
            features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        if not is_training:
            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
            all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask
            )
            return features, dataset
        else:
            ##初始化数据结构
            new_features=[]
            example_index_to_features = collections.defaultdict(list)#建立字典
            for feature in features:
                example_index_to_features[feature.example_index].append(feature)

            all_input_ids = []
            all_attention_masks = []
            all_token_type_ids = []
            all_start_positions = []
            all_end_positions = []
            all_qve_input_ids = []
            all_qve_attention_masks = []
            all_qa_qve_attention_masks = []
            all_ca_qve_attention_masks = []
            all_qc_qve_attention_masks = []
            all_qve_token_type_ids = []
            special_tokens_ids = tokenizer.additional_special_tokens_ids

            #特征处理
            count = 0

            for ex, efs in example_index_to_features.items():
                tmp_feature = None
                for ef in efs:#找出最合适的特征
                    if ef.start_position > 0 and ef.end_position > 0:
                        if tmp_feature:
                            if ef.start_position < tmp_feature.start_position:
                                tmp_feature = ef
                        else:
                            tmp_feature = ef

                if not tmp_feature:
                    tmp_feature = efs[0]

                #计算问题和段落特征以便进行重新编码
                question_len = tmp_feature.input_ids.index(tokenizer.sep_token_id) + 1
                new_para_ids = tmp_feature.tokens[question_len:-1]

                new_question_ids = tmp_feature.input_ids[1:question_len - 1]

                if tmp_feature.start_position == tmp_feature.end_position == 0:
                    new_question_ids = new_question_ids + [special_tokens_ids[0]] + [special_tokens_ids[1]]
                else:
                    new_question_ids = new_question_ids + [special_tokens_ids[0]] + \
                                       tmp_feature.input_ids[tmp_feature.start_position:tmp_feature.end_position + 1]
                                

                #创建QVE模型输入
                encoded_dict = tokenizer.encode_plus(
                    new_question_ids,
                    new_para_ids,
                    truncation=True,
                    padding=padding_strategy,
                    max_length=max_seq_length,
                    return_token_type_ids=True,
                )

                qve_input_id = encoded_dict.data['input_ids']
                qve_attention_mask = encoded_dict.data['attention_mask']
                qve_token_type_id = encoded_dict.data['token_type_ids']
               
                if tmp_feature.start_position==0 and tmp_feature.end_position==0:
                    continue

                orig_question_len = question_len - 2  # 减去 [CLS] 和第一个 [SEP]
                answer_len = tmp_feature.end_position - tmp_feature.start_position + 1

                full_attention_mask = encoded_dict['attention_mask']
                seq_length = len(full_attention_mask)

                # 问题部分起始位置：1 ([CLS]之后)
                question_start = 1
                # 问题部分结束位置：问题长度 + 1
                question_end = question_len
                # 答案起始位置：问题结束位置 + 1
                answer_start = question_end + 1
                # 答案结束位置：答案起始位置 + 答案长度 - 1
                answer_end = answer_start + answer_len - 1
                # 段落起始位置：答案结束位置 + 1
                para_start = answer_end + 1

                qa_attention_mask = full_attention_mask.copy()
                if para_start < seq_length:
                    qa_attention_mask[para_start:] = [0] * (seq_length - para_start)

                ca_attention_mask = full_attention_mask.copy()
                ca_attention_mask[question_start:question_end] = [0] * (question_end - question_start)
                ca_attention_mask[answer_start:answer_end+1] = full_attention_mask[answer_start:answer_end+1]

                qc_attention_mask = full_attention_mask.copy()
                if answer_end < seq_length:
                    qc_attention_mask[answer_start:answer_end+1] = [0] * (answer_end - answer_start + 1)

                # if count <= 3 and (tmp_feature.start_position != 0 or tmp_feature.end_position != 0):
                #     logger.info("===== Sample %d =====", count + 1)
                    
                #     # 1. 解码原始输入部分
                #     logger.info("1. Decoded Input Parts:")
                #     logger.info("   Question: %s", tokenizer.decode(tmp_feature.input_ids[1:question_len - 1], skip_special_tokens=False))
                #     logger.info("   Answer: %s", tokenizer.decode(
                #         tmp_feature.input_ids[tmp_feature.start_position:tmp_feature.end_position + 1],
                #         skip_special_tokens=False
                #     ))
                #     logger.info("   Paragraph: %s", tokenizer.convert_tokens_to_string(new_para_ids))
                    
                #     # 2. 解码encoded_dict的内容
                #     logger.info("2. Full Combined Sequence from encoded_dict:")
                #     full_input = tokenizer.decode(encoded_dict['input_ids'])
                #     logger.info("   Full Text: %s", full_input)
                    
                #     # 3. 显示与attention mask一一对应的token形式
                #     logger.info("3. Tokens with Attention Masks (aligned):")
                #     tokens = tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'])
                #     token_lengths = [len(token) for token in tokens]
                #     max_length = max(token_lengths) + 1  # 为padding留出空间
                    
                #     # 创建带对齐的token显示
                #     aligned_tokens = []
                #     for token, length in zip(tokens, token_lengths):
                #         padding = " " * (max_length - length)
                #         aligned_tokens.append(token + padding)
                    
                #     logger.info("   Tokens:  %s", " ".join(aligned_tokens))
                #     logger.info("   Base Mask:%s", " ".join([f"{mask:^{max_length}}" for mask in encoded_dict['attention_mask']]))
                #     logger.info("   QA Mask:  %s", " ".join([f"{mask:^{max_length}}" for mask in qa_attention_mask]))
                #     logger.info("   CA Mask:  %s", " ".join([f"{mask:^{max_length}}" for mask in ca_attention_mask]))
                #     logger.info("   QC Mask:  %s", " ".join([f"{mask:^{max_length}}" for mask in qc_attention_mask]))
                    
                #     logger.info("=" * 50)
                #     count += 1

                 #创建PyTorch张量
                all_qve_input_ids.append(qve_input_id)
                all_qve_attention_masks.append(qve_attention_mask)
                all_qa_qve_attention_masks.append(qa_attention_mask)
                all_ca_qve_attention_masks.append(ca_attention_mask)
                all_qc_qve_attention_masks.append(qc_attention_mask)
                all_qve_token_type_ids.append(qve_token_type_id)

                all_input_ids.append(tmp_feature.input_ids)
                all_attention_masks.append(tmp_feature.attention_mask)
                all_token_type_ids.append(tmp_feature.token_type_ids)
                all_start_positions.append(tmp_feature.start_position)
                all_end_positions.append(tmp_feature.end_position)
                new_features.append(tmp_feature)


            all_qve_input_ids = torch.tensor(all_qve_input_ids, dtype=torch.long)
            all_qve_attention_masks = torch.tensor(all_qve_attention_masks, dtype=torch.long)
            all_qa_qve_attention_masks = torch.tensor(all_qa_qve_attention_masks, dtype=torch.long)
            all_ca_qve_attention_masks = torch.tensor(all_ca_qve_attention_masks, dtype=torch.long)
            all_qc_qve_attention_masks = torch.tensor(all_qc_qve_attention_masks, dtype=torch.long)
            all_qve_token_type_ids = torch.tensor(all_qve_token_type_ids, dtype=torch.long)

            all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
            all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
            all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
            all_start_positions = torch.tensor(all_start_positions, dtype=torch.long)
            all_end_positions = torch.tensor(all_end_positions, dtype=torch.long)

            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_feature_index,
                all_start_positions,
                all_end_positions,
                all_qve_input_ids,
                all_qve_attention_masks,
                all_qa_qve_attention_masks,
                all_ca_qve_attention_masks,
                all_qc_qve_attention_masks,
                all_qve_token_type_ids,
            )
            features = new_features
            return features, dataset
    elif return_dataset == "tf":
        if not is_tf_available():
            raise RuntimeError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for i, ex in enumerate(features):
                if ex.token_type_ids is None:
                    yield (
                        {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "feature_index": i,
                            "qas_id": ex.qas_id,
                        },
                        {
                            "start_positions": ex.start_position,
                            "end_positions": ex.end_position,
                            "cls_index": ex.cls_index,
                            "p_mask": ex.p_mask,
                            "is_impossible": ex.is_impossible,
                        },
                    )
                else:
                    yield (
                        {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                            "feature_index": i,
                            "qas_id": ex.qas_id,
                        },
                        {
                            "start_positions": ex.start_position,
                            "end_positions": ex.end_position,
                            "cls_index": ex.cls_index,
                            "p_mask": ex.p_mask,
                            "is_impossible": ex.is_impossible,
                        },
                    )

        # Why have we split the batch into a tuple? PyTorch just has a list of tensors.
        if "token_type_ids" in tokenizer.model_input_names:
            train_types = (
                {
                    "input_ids": tf.int32,
                    "attention_mask": tf.int32,
                    "token_type_ids": tf.int32,
                    "feature_index": tf.int64,
                    "qas_id": tf.string,
                },
                {
                    "start_positions": tf.int64,
                    "end_positions": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            )

            train_shapes = (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                    "feature_index": tf.TensorShape([]),
                    "qas_id": tf.TensorShape([]),
                },
                {
                    "start_positions": tf.TensorShape([]),
                    "end_positions": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            )
        else:
            train_types = (
                {"input_ids": tf.int32, "attention_mask": tf.int32, "feature_index": tf.int64, "qas_id": tf.string},
                {
                    "start_positions": tf.int64,
                    "end_positions": tf.int64,
                    "cls_index": tf.int64,
                    "p_mask": tf.int32,
                    "is_impossible": tf.int32,
                },
            )

            train_shapes = (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "feature_index": tf.TensorShape([]),
                    "qas_id": tf.TensorShape([]),
                },
                {
                    "start_positions": tf.TensorShape([]),
                    "end_positions": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                    "is_impossible": tf.TensorShape([]),
                },
            )

        return tf.data.Dataset.from_generator(gen, train_types, train_shapes)
    else:
        return features


class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set.
    Overriden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and version 2.0 of SQuAD, respectively.
    """

    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        return SquadExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of :class:`~transformers.data.processors.squad.SquadExample` using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
            evaluate: boolean specifying if in evaluation mode or in training mode

        Returns:
            List of SquadExample

        Examples::

            >>> import tensorflow_datasets as tfds
            >>> dataset = tfds.load("squad")

            >>> training_examples = get_examples_from_dataset(dataset, evaluate=False)
            >>> evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        """

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in tqdm(dataset):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
                os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, filename=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
                os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    is_impossible = qa.get("is_impossible", False)
                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    examples.append(example)
        return examples


class SquadV1Processor(SquadProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"


class SquadV2Processor(SquadProcessor):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"


class SquadExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
            self,
            qas_id,
            question_text,
            context_text,
            answer_text,
            start_position_character,
            title,
            answers=[],
            is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]


class SquadFeatures:
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            cls_index,
            p_mask,
            example_index,
            unique_id,
            paragraph_len,
            token_is_max_context,
            tokens,
            token_to_orig_map,
            start_position,
            end_position,
            is_impossible,
            qas_id: str = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id


class SquadResult:
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits
