import json_lines  # 导入用于解析JSON Lines格式数据的库
import json  # 导入JSON库，用于处理JSON数据
from argparse import ArgumentParser  # 导入命令行参数解析库

# 初始化数据存储和计数变量
data = []
note = 0
count = 0

# 设置命令行参数解析器
parser = ArgumentParser()
# 添加输入文件参
parser.add_argument("-input", dest="input", default="TriviaQA-web.jsonl")
# 添加输出文件参数
parser.add_argument("-output", dest="output", default="TriviaQA-web.json")
args = parser.parse_args()

# 打开输入和输出文件
with open(args.input, 'rb') as f, open(args.output, "w") as out:
    # 逐行读取JSON Lines格式的数据，存入data列表
    for item in json_lines.reader(f):
        data.append(item)

    # 创建新的数据结构，用于格式化输出，设定版本信息
    new_data = {'data': [], 'version': 'v0.1'}

    # 设置初始数据起始索引，如果第一项是header，则从第二项开始遍历
    starter = 0
    if 'header' in data[0]:
        starter = 1
    else:
        data = data[0]

    # 遍历所有数据项，从starter索引开始
    for i in range(starter, len(data)):
        # 创建新的条目，包含title（当前索引）和空的段落列表
        new_data['data'].append({'title': str(i), 'paragraphs': []})

        # 获取当前项的上下文文本
        context = data[i]['context']
        len1=len(context)# 保存原始上下文长度，用于后续校验
        # 替换上下文中的特殊空格字符
        context = context.replace(u'\u00A0', ' ')
        assert len(context)==len1# 确认替换后长度不变

         # 添加上下文和问答列表到段落
        new_data['data'][note]['paragraphs'].append({'context': context, 'qas': []})

        # 遍历当前项的所有问答
        for k in range(len(data[i]['qas'])):
            answers = []# 收集有效答案

            # 遍历检测到的答案
            for p in range(len(data[i]['qas'][k]['detected_answers'])):
                text = data[i]['qas'][k]['detected_answers'][p]['text']
                answer_index = data[i]['qas'][k]['detected_answers'][p]['char_spans'][0]
                # 提取并格式化答案信息
                answers.append({'answer_start': answer_index[0], 'text': context[answer_index[0]:answer_index[1]+1]})
            # 如果有有效答案
            if len(answers) > 0:
                # 添加问题、问题ID和答案到问答列表
                new_data['data'][note]['paragraphs'][0]['qas'].append(
                    {'question': data[i]['qas'][k]['question'], 'id': data[i]['qas'][k]['qid'], 'answers': answers})
                count += 1
        note += 1

    print(count)
    new_data["len"] = count
    json.dump(new_data, out)
