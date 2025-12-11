# 导入所需的库
import json  # 用于处理JSON数据格式
import random  # 用于生成随机数
from argparse import ArgumentParser  # 用于处理命令行参数
from numpy.random import default_rng  # 从NumPy导入默认随机数生成器

parser = ArgumentParser()
# 添加命令行参数，包括输入文件、输出文件、抽样数量和随机种子
parser.add_argument("--in_file", type=str, default="data/NewsQA.train.json",)
parser.add_argument("--out_file_dev", type=str,  default="dataNewsQA.sample.dev.json")
parser.add_argument("--out_file_train", type=str, default="data/NewsQA.sample.train.json")
parser.add_argument("--num", type=int, default=1000, required=False)
parser.add_argument("--seed", type=int, default=42, required=False)

args = parser.parse_args()

# 定义函数以随机方式从数据集中抽样
def subsample_dataset_random(data_json, sample_num=1000, seed=55):

    #初始化
    total = 0
    context_num=0
    id_lists=[]

     # 遍历数据集以计算总QA数和上下文数，并收集QA ID
    for paras in data_json['data']:
        for para in paras['paragraphs']:
            context_num+=1
            qa_num = len(para['qas'])
            id_lists+=[qa['id'] for qa in para['qas']]
            total += qa_num
    # 打印总的QA数和上下文数
    print('Total QA Num: %d, Total Context: %d' % (total,context_num))

    random.seed(seed) # 设置随机种子
    rng = default_rng()# 创建随机数生成器对象
    # 从QA ID列表中随机选择指定数量的ID，组合成新的抽样列表
    sampled_list = list(rng.choice(id_lists, size=sample_num,replace=False))
    new_passages_dev = [] # 用于存储Dev集的段落
    new_passages_train=[] # 用于存储Train集的段落

    # 遍历数据以重新组织成Dev和Train集的数据结构
    for passages in data_json['data']:
        new_paras_dev = []  # 用于存储Dev集的段落
        new_paras_train = []  # 用于存储Train集的段落

        for para in passages['paragraphs']:
            context = para['context']# 获取上下文
            new_qas_dev = []    # 用于存储Dev集的QA
            new_qas_train = [] # 用于存储Train集的QA

            # 根据抽样列表将QA分配给Dev集或Train集
            for qa in para['qas']:
                if qa['id'] in sampled_list:
                    new_qas_dev.append(qa)
                else:
                    new_qas_train.append(qa)

            # 如果有Dev集的QA，则添加到新的Dev集段落
            if len(new_qas_dev) > 0:
                new_paras_dev.append({'context': context, 'qas': new_qas_dev})
            # 如果有Train集的QA，则添加到新的Train集段落
            if len(new_qas_train) > 0:
                new_paras_train.append({'context': context, 'qas': new_qas_train})

        # 如果有Dev集的段落，则添加到新的Dev集
        if len(new_paras_dev) > 0:
            new_passages_dev.append({'title': passages['title'], 'paragraphs': new_paras_dev})

        # 如果有Train集的段落，则添加到新的Train集
        if len(new_paras_train) > 0:
            new_passages_train.append({'title': passages['title'], 'paragraphs': new_paras_train})

    # 创建新的JSON结构体用于Dev集和Train集
    dev_data_json = {'data': new_passages_dev, 'version': data_json['version']}
    train_data_json = {'data': new_passages_train, 'version': data_json['version']}

    # 计算Dev集的总QA数和上下文数，并打印
    total = 0
    context_num=0
    for paras in dev_data_json['data']:
        for para in paras['paragraphs']:
            context_num+=1
            qa_num = len(para['qas'])
            id_lists+=[qa['id'] for qa in para['qas']]
            total += qa_num
    print('Sample Dev QA Num: %d, Total Context: %d' % (total,context_num))

    # 计算Train集的总QA数和上下文数，并打印
    total = 0
    context_num = 0
    for paras in train_data_json['data']:
        for para in paras['paragraphs']:
            context_num += 1
            qa_num = len(para['qas'])
            id_lists += [qa['id'] for qa in para['qas']]
            total += qa_num
    print('Sample Train QA Num: %d, Total Context: %d' % (total, context_num))

    return train_data_json,dev_data_json

# 主函数
def main(args):

    # 读取数据集
    dataset = json.load(open(args.in_file, 'r'))

    # 调用抽样函数，得到抽样后的Train集和Dev集
    train_data_json,dev_data_json=subsample_dataset_random(dataset, args.num, args.seed)

    # 将抽样后的数据集保存到指定的文件中
    json.dump(train_data_json, open(args.out_file_train, 'w'))
    json.dump(dev_data_json, open(args.out_file_dev, 'w'))


if __name__ == '__main__':
    args = parser.parse_args()
    #设置随机种子
    random.seed(args.seed)
    main(args)
