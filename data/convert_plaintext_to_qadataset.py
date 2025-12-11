import argparse
import json
from spacy.lang.en import English

# 初始化spacy的英语模型和分词器（注：分词器在代码中未实际使用，可能是冗余代码）
nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

# 设置命令行参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--input_generation_file', type=str, required=True,help='input_source_file')
parser.add_argument('--input_id_file', type=str, required=True,help='input_id_file')
parser.add_argument('--input_qa_data_file', type=str, required=True,help='input_qa_data_file')
parser.add_argument('--output_file', type=str,required=True, help='output_file')
parser.add_argument('--percentage', type=float,default=1.0, help='filter percentage')



def convert_plaintext2qadataset(input_generation_file,input_qa_data_file,input_id_file,output_file,percentage):
    # 读取ID文件和生成的问题+得分文件
    generations=[]
    ids=[]

    # 读取ID列表
    with open(input_id_file) as rf:
        for line in rf:
            ids.append(line.strip())# 去除每行的前后空格
        rf.close()

    # 读取生成的问题和得分（格式：问题文本\t得分）
    with open(input_generation_file) as rf:
        for line in rf:
            content = line.strip().split("\t")
            assert len(content)==2# 确保每行包含问题和得分
            generation=content[0]
            lm_score=float(content[1])
            generations.append((generation,lm_score))
        rf.close()

    assert len(generations)==len(ids)

    # 确保ID和生成问题数量一致
    id2generationwithscore={ids[i]:generations[i] for i in range(len(generations))}
     # 确定得分阈值
    if percentage==1.0:
        threshold=-10000000
    else:
        # 按得分降序排序
        sorted_d=sorted(id2generationwithscore.items(), key=lambda x: x[1][1], reverse=True)
        # 计算保留百分比对应的阈值位置
        threshold=id2generationwithscore[sorted_d[int(percentage*len(sorted_d))][0]][1]# 可能存在保留数量多于预期的边界问题

    # 加载原始QA数据
    datajson = json.load(open(input_qa_data_file, 'r'))
    count=0
    new_passages = []
    # 遍历每个段落，进行问题替换和过滤
    for passages in datajson['data']:
        new_paras = []
        for para in passages['paragraphs']:
            context = para['context']
            new_qas = []
            for qa in para['qas']:
                # 检查当前问题的得分是否达到阈值
                if id2generationwithscore[qa['id']][1] >= threshold:
                    qa['question']=id2generationwithscore[qa['id']][0]
                    new_qas.append(qa)
                    count+=1
            # 仅保留包含有效问题的段落
            if len(new_qas) > 0:
                new_paras.append({'context': context, 'qas': new_qas})
        if len(new_paras) > 0:
            new_passages.append({'title': passages['title'], 'paragraphs': new_paras})
     # 统计并打印过滤前后的数据量对比
    print(count)
    # double-check

    # 原始数据统计
    qa_num = 0
    context_num = 0
    for passages in datajson['data']:
        for para in passages['paragraphs']:
            context_num += 1
            for qa in para['qas']:
                if len(qa['answers']) > 0:
                    qa_num += len(qa['answers'])

    print("Before filtering: #Context: %d , #QA: %d" % (context_num, qa_num))

    # 更新数据并重新统计
    datajson['data'] = new_passages

    qa_num = 0
    context_num = 0
    for passages in datajson['data']:
        for para in passages['paragraphs']:
            context_num += 1
            for qa in para['qas']:
                if len(qa['answers']) > 0:
                    qa_num += len(qa['answers'])

    print("After filtering: #Context: %d , #QA: %d" % (context_num, qa_num))

    # 保存处理后的数据
    json.dump(datajson,open(output_file,'w'))

# 主程序入口
if __name__=="__main__":
    args = parser.parse_args()
    convert_plaintext2qadataset(args.input_generation_file, args.input_qa_data_file, args.input_id_file, args.output_file,args.percentage)