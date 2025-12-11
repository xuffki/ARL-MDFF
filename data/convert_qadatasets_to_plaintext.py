import argparse
import json
from tqdm import tqdm
import random
import re
from transformers import BartTokenizer



parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True,help='input_file')
parser.add_argument('--output_srcfile', type=str,required=True, help='output_srcfile')
parser.add_argument('--output_tgtfile', type=str,required=True, help='output_tgtfile')
parser.add_argument('--output_idfile', type=str,required=True, help='output_idfile')
parser.add_argument('--max_source_length', type=int, default=512)
parser.add_argument('--min_source_length', type=int, default=1)
parser.add_argument('--doc_stride', type=int, default=128)
parser.add_argument('--tokenizer_path', type=str, required=True)


def chunk_stride(lst, n, stride):
    yield_flag = True
    for i in range(0, len(lst), stride):
        if yield_flag:
            if i + n >= len(lst):# 当剩余部分不足n时停止
                yield_flag = False
            yield lst[i: i + n]# 返回长度为n的分块


def convert_squad2plaintext(input_file,output_srcfile, output_tgtfile, output_idfile,tokenizer):
    srcs=[]
    tgts=[]
    ids=[]
    #读取输入数据
    data_json= json.load(open(input_file, 'r'))
    for paras in tqdm(data_json['data']):  # 遍历所有段落
        for para in paras['paragraphs']:
            context = para['context']      # 当前段落文本
            for qa in para['qas']:         # 遍历所有问题
                question = qa['question']
                answer = qa['answers'][0]  # 取第一个答案
                #标记答案位置
                answer_start = answer['answer_start']
                answer_end = answer_start + len(answer['text'])
                #用特殊标记<hl>包裹答案
                tag_context = context[:answer_start] + " <hl> " + \
                              context[answer_start:answer_end] \
                              + " <hl> " + context[answer_end:]

                # 文本清理
                # 合并多余空格
                tag_context = " ".join(tag_context.split())
                question = " ".join(question.split())

                #分词与长度处理
                src = tokenizer.encode(tag_context, add_special_tokens=False, max_length=100000, truncation=True)
                max_len = args.max_source_length - 2 # 预留位置给特殊标记[CLS]/[SEP]

                if len(src) < args.min_source_length-2:
                    continue # 过滤过短文本

                if len(src) > max_len:
                    # 确保分块包含答案位置
                    # ensure that src includes answer
                    ans_end_index = src.index(tokenizer.additional_special_tokens_ids[0]) # 找到<hl>的位置
                    assert ans_end_index >= 0
                    for jj, con_chunk in enumerate(chunk_stride(src, max_len, args.doc_stride)):
                        if jj * args.doc_stride + max_len > ans_end_index: # 选择包含答案的分块
                            src = con_chunk
                            break

                tag_context=tokenizer.decode(src)# 将分词ID转回文本

                tag_context = "generate question: " + tag_context# 添加任务提示
                srcs.append(tag_context)# 源文本（带高亮的上下文）
                tgts.append(question)# 目标文本（问题）
                ids.append(qa['id'])# 问题ID


    assert len(srcs) == len(tgts)
    print(len(srcs))

    with open(output_srcfile, 'w') as fout:# 写入源文本
        for src in srcs:
            fout.write(src + "\n")
        fout.close()

    with open(output_tgtfile, 'w') as fout:# 写入目标文本
        for tgt in tgts:
            fout.write(tgt + "\n")
        fout.close()

    with open(output_idfile, 'w') as fout:# 写入问题ID
        for qid in ids:
            fout.write(qid + "\n")
        fout.close()


if __name__=="__main__":
    args = parser.parse_args()
    random.seed(1)# 固定随机种子

    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)
    special_tokens_dict = {'additional_special_tokens': ['<hl>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    convert_squad2plaintext(args.input_file,args.output_srcfile,args.output_tgtfile,args.output_idfile,tokenizer)

"""
将SQuAD格式数据集转化为
"data": [{
  "paragraphs": [{
    "context": "Alice was born in Paris. She works at Google.",
    "qas": [{
      "question": "Where does Alice work?",
      "id": "1",
      "answers": [{"text": "Google", "answer_start": 24}]
    }]
  }]
}]
形式
"""

