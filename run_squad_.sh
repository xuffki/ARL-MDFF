#!/bin/bash

# 定义基础变量
dataset_name="NaturalQuestionsShort"
#path="checkpoints/NaturalQuestionsShort_QVE_1000_base"
path="checkpoints/NaturalQuestionsShort_QVE_500_base"

# 训练文件列表
train_files=(
    # "filtered_score_ca.json"
    # "filtered_score_qc.json"
    # "filtered_score_qa.json"
    # "filtered_score_qc_qa.json"
    # "filtered_score_ca_qa.json"
    # "filtered_score_ca_qc.json"
    "filtered_score_QVE.json"
    # "filtered_score_0.80_0.10_0.10.json"
    # "filtered_score_0.10_0.80_0.10.json"
    # "filtered_score_0.10_0.10_0.80.json"
    # "filtered_score_0.33_0.33_0.34.json"  
)

# 遍历所有训练文件
for train_file in "${train_files[@]}"; do
    # 提取文件标识（用于输出目录命名）
    suffix="${train_file#"filtered_score_"}"
    suffix="${suffix%.*}"
    
    # 构建输出目录路径
    output_dir="checkpoints/QA_${dataset_name}_Source_Sythetic_QVEFiltering_${suffix}"
    
    # 打印当前任务信息
    echo "==================================================="
    echo "Processing training file: ${train_file}"
    echo "Output will be saved to: ${output_dir}"
    echo "==================================================="
    
    # 执行Python命令
    python QA/run_squad.py \
        --model_type bert \
        --model_name_or_path checkpoints/QA_source_only/ \
        --do_train \
        --do_eval \
        --do_lower_case \
        --train_file "${path}/${train_file}" \
        --predict_file "data/${dataset_name}.test.json" \
        --per_gpu_train_batch_size 12 \
        --learning_rate 3e-5 \
        --num_train_epochs 1.0 \
        --max_seq_length 384 \
        --threads 24 \
        --per_gpu_eval_batch_size 32 \
        --doc_stride 128 \
        --output_dir "${output_dir}" \
        --save_steps 20000 \
        --overwrite_cache \
        --overwrite_output_dir
        
    # 每次运行后添加空行便于阅读
    echo -e "\n\n"
done

echo "All training jobs completed!"