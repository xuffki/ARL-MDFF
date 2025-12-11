#! /bin/bash

cd data

# 定义一个数组，其中包含五个数据集的名称
#declare -a arr=("SQuAD" "NewsQA" "NaturalQuestionsShort" "HotpotQA" "TriviaQA-web")
declare -a arr=("SQuAD" "NewsQA" "NaturalQuestionsShort" "HotpotQA" "TriviaQA-web")

##downloading the datasets from the MRQA 2019.
##We use the dev set as the test set

# 循环遍历数组中的每个数据集名称
for dataset_name in "${arr[@]}"; do
  # 输出正在下载的数据集名称
#   echo "Downloading dataset: $dataset_name ..."
#   # 使用wget命令下载训练集，保存为.gz压缩文件
#   wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/"$dataset_name".jsonl.gz -O "$dataset_name".train.jsonl.gz
#   # 使用wget命令下载开发集（用作测试集），保存为.gz压缩文件
#   wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/"$dataset_name".jsonl.gz -O "$dataset_name".test.jsonl.gz

# # 解压训练集的.gz文件，生成.jsonl文件
#   gzip -d "$dataset_name".train.jsonl.gz

#   # 使用Python脚本将.jsonl格式转换为类似SQuAD格式的.json文件
#   python convert_jsonl2json.py \
#   -input "$dataset_name".train.jsonl \
#   -output "$dataset_name".train.json

#   # 解压测试集的.gz文件，生成.jsonl文件
#   gzip -d "$dataset_name".test.jsonl.gz
#   # 使用Python脚本将.jsonl格式转换为类似SQuAD格式的.json文件
#   python convert_jsonl2json.py \
#   -input "$dataset_name".test.jsonl \
#   -output "$dataset_name".test.json

#   # 删除已经转换完的.jsonl文件，节省空间
#   rm "$dataset_name".train.jsonl
#   rm "$dataset_name".test.jsonl

  ##For all the target domain datasets {"NewsQA" "NaturalQuestionsShort" "HotpotQA" "TriviaQA-web"}
  ##We sample 1000 QAs from the training as the dev set
  # 注释：对所有目标领域的数据集 {"NewsQA" "NaturalQuestionsShort" "HotpotQA" "TriviaQA-web"} 从训练集中抽样1000条问答作为开发集。
  if [ "$dataset_name" != "SQuAD" ]; then
    echo "Sampling dev set from the training set..."
    # 使用Python脚本从训练集抽样1000条数据作为开发集，修改后的训练集文件也保存
    python split_data_num.py \
    --in_file "$dataset_name".train.json \
    --out_file_dev "$dataset_name".sample.dev.json \
    --out_file_train "$dataset_name".sample.train.json \
    --num 500

    # echo "Sampling dev sets with different sizes from $dataset_name training set..."
    # for sample_size in 500 1000 5000 10000; do
    #   echo "Sampling $sample_size examples..."
    #   python split_data_num.py \
    #     --in_file "$dataset_name".train.json \
    #     --out_file_dev "$dataset_name".sample.$sample_size.dev.json \
    #     --out_file_train "$dataset_name".sample.$sample_size.train.json \
    #     --num $sample_size
    # done
  fi
done
