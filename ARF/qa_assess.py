import os  
import argparse  
import logging  
import random  
import json  
import copy  
import warnings  
from collections import Counter  

import torch  
import numpy as np  
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset  
from torch.distributions.binomial import Binomial  

from tqdm import tqdm, trange  

from transformers import AutoTokenizer, AutoConfig, BertForQuestionAnswering  

from squad_processing import squad_convert_examples_to_features
from modeling_bert import BertForSequenceClassification,BertForQuestionAnswering
from transformers.data.processors.squad import SquadV1Processor

logger = logging.getLogger(__name__)

PT_LR_SCHEDULER_WARNING = "Please also save or load the state of the optimzer when saving or loading the scheduler."

def calculate_marginal_info(qa_model_marginal, train_dataset, args):
    # calculate marginal information
    if args.n_gpu > 0:
        qa_model_marginal=torch.nn.DataParallel(qa_model_marginal)
    args.dev_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.dev_batch_size)
    qa_model_marginal.eval()
    marginal_info_start=[]
    marginal_info_end=[]

    for train_batch in tqdm(train_dataloader):
        train_batch = tuple(t.to(args.device) for t in train_batch)
        with torch.no_grad():
            inputs = {
                "input_ids": train_batch[0],
                "attention_mask": train_batch[1],
                "token_type_ids": train_batch[2],
            }

            outputs = qa_model_marginal(**inputs)
            pred_start = torch.sigmoid(outputs[0])
            pred_end = torch.sigmoid(outputs[1])
            true_start = torch.nn.functional.one_hot(train_batch[4], pred_start.shape[1])
            true_end = torch.nn.functional.one_hot(train_batch[5],pred_end.shape[1])
            pred_start_prob = (true_start * pred_start).sum(dim=1)
            pred_end_prob = (true_end * pred_end).sum(dim=1)
            diff_start = torch.abs(1 - pred_start_prob)
            diff_end = torch.abs(1 - pred_end_prob)
            marginal_info_start.append(diff_start)
            marginal_info_end.append(diff_end)

    marginal_info_start = torch.cat(marginal_info_start).view(-1,1)
    marginal_info_end = torch.cat(marginal_info_end).view(-1,1)
    marginal_info=torch.cat((marginal_info_start,marginal_info_end),dim=1)

    return marginal_info

def load_and_cache_examples(args, tokenizer, qa_model_marginal=None, output_examples=False, dev=False):

    # Load data features from cache or dataset file
    mode="dev" if dev else "train"

    qa_dataset_name = [n for n in ["NewsQA","NaturalQuestionsShort","HotpotQA","TriviaQA-web"] if n in args.train_file][0]
    cached_features_file = os.path.join(
        "cached_{}_{}_{}".format(
            mode,
            qa_dataset_name,
            str(args.max_seq_length),
        ),
    )
    # cached_features_file = os.path.join(
    #     "cached_qa_{}_{}_{}".format(
    #         mode,
    #         qa_dataset_name,
    #         str(args.max_seq_length),
    #     ),
    # )
    

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        process_file = args.dev_file if dev else args.train_file
        logger.info("Creating features from dataset file at: %s", process_file)

        processor = SquadV1Processor()
        examples = processor.get_train_examples(data_dir=None, filename=process_file)

        for example in examples[:3]:  # 检查前3个样本
            logger.info(f"问题ID: {example.qas_id}")
            logger.info(f"问题: {example.question_text}")
            logger.info(f"答案（新context）: {example.context}\n{'-'*50}")   


        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True,
            return_dataset="pt",
            threads=args.threads,
        )

        sample_input = tokenizer.decode(features[0].input_ids)
        logger.info(f"输入序列示例：\n{sample_input}")

        if qa_model_marginal and not dev:
            marginal_info = calculate_marginal_info(qa_model_marginal, dataset, args)
            new_data = dataset.tensors + (marginal_info.cpu(),)
            dataset = TensorDataset(*new_data)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset

def score_samples(args):
    """对数据集前N个样本进行边际信息评分"""
    # 初始化设备和配置
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.device = device

    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_path,
        do_lower_case=args.do_lower_case
    )
    
    config = AutoConfig.from_pretrained(args.model_path)
    model = BertForQuestionAnswering.from_pretrained(args.model_path, config=config)
    model.to(args.device)

    # 加载数据集（需要原始示例信息）
    dataset, examples, features = load_and_cache_examples(
        args, 
        tokenizer,
        output_examples=True,
        dev=False  # 使用训练集或开发集根据需求调整
    )

    # 计算所有样本的边际信息
    marginal_info = calculate_marginal_info(model, dataset, args)
    
    # 显示前N个样本的详细信息
    print(f"\n{'='*50}")
    print(f"Displaying marginal scores for first {args.num_samples} samples:")
    print(f"{'='*50}\n")
    
    for i in range(min(args.num_samples, len(examples))):
        example = examples[i]
        scores = marginal_info[i]

        # 获取答案上下文（截断显示）
        context = example.context_text[:200] + "..." if len(example.context_text) > 200 else example.context_text
        
        print(f"Sample #{i+1}")
        print(f"Question ID: {example.qas_id}")
        print(f"Question: {example.question_text}")
        print(f"Answer: {example.answer_text}")
        print(f"Context: {context}")
        print(f"Start Position Score: {scores[0].item():.4f}")
        print(f"End Position Score: {scores[1].item():.4f}")
        print(f"{'-'*50}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 必需参数
    parser.add_argument("--model_path", 
                      type=str, 
                      required=True,
                      help="预训练QA模型路径")
    parser.add_argument("--train_file",
                      type=str,
                      required=True,
                      help="输入训练文件路径")
    
    # 新增参数
    parser.add_argument("--tokenizer_name",
                      type=str,
                      default="",
                      help="分词器名称/路径（默认使用model_path）")
    parser.add_argument("--num_samples",
                      type=int,
                      default=5,
                      help="需要显示的样本数量（默认：5）")
    parser.add_argument("--per_gpu_eval_batch_size",
                      default=128,
                      type=int,
                      help="评估批次大小（默认：128）")
    
    # 从原始代码继承的必要参数
    parser.add_argument("--max_seq_length",
                      default=384,
                      type=int,
                      help="最大序列长度（默认：384）")
    parser.add_argument("--doc_stride",
                      default=128,
                      type=int,
                      help="文档滑动窗口步长（默认：128）")
    parser.add_argument("--do_lower_case",
                      action="store_true",
                      help="使用小写分词")
    parser.add_argument("--no_cuda",
                      action="store_true",
                      help="禁用CUDA")
    parser.add_argument("--overwrite_cache",
                      action="store_true",
                      help="覆盖缓存文件")
    parser.add_argument("--threads",
                      type=int,
                      default=24,
                      help="多线程处理数（默认：24）")
    
    args = parser.parse_args()
    
    # 自动填充缺失的tokenizer_name
    if not args.tokenizer_name:
        args.tokenizer_name = args.model_path
    
    # 运行评分
    score_samples(args)