# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD."""

import warnings
import argparse
import logging
import os
import random
from torch.distributions.binomial import Binomial
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
import copy
import json

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from squad_processing import squad_convert_examples_to_features,squad_convert_3p_examples_to_features
from modeling_bert import BertForSequenceClassification,BertForQuestionAnswering, DynamicWeightModel
from transformers.data.processors.squad import SquadV1Processor

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

PT_LR_SCHEDULER_WARNING = "Please also save or load the state of the optimzer when saving or loading the scheduler."


def reissue_pt_warnings(caught_warnings):
    # Reissue warnings that are not the PT_LR_SCHEDULER_WARNING
    if len(caught_warnings) > 1:
        for w in caught_warnings:
            if w.category != UserWarning or w.message != PT_LR_SCHEDULER_WARNING:
                warnings.warn(w.message, w.category)


def set_seed(args):
    #设置全局随机种子​
    random.seed(args.seed)             # Python内置随机模块
    np.random.seed(args.seed)          # NumPy随机生成器
    torch.manual_seed(args.seed)       # PyTorch CPU随机种子
    if args.n_gpu > 0:                # 如果使用GPU
        torch.cuda.manual_seed_all(args.seed)  # 所有GPU的随机种子


def to_list(tensor):
    #将PyTorch张量转换为Python列表​
    return tensor.detach().cpu().tolist()


def calculate_marginal_info(qa_model_marginal, train_dataset, args):
    """计算问答模型的边际信息（用于主动学习等场景）
    
    参数：
        qa_model_marginal : transformers模型
            用于计算边际信息的问答模型，应支持输出start_logits和end_logits
        train_dataset : Dataset
            需要计算边际信息的训练数据集
        args : 参数对象
            包含以下属性：
            - n_gpu: 可用GPU数量
            - per_gpu_eval_batch_size: 单GPU评估批大小
            - device: 计算设备（如cuda或cpu）
    
    返回：
        torch.Tensor: 形状为[N, 2]的边际信息张量，
        每行包含对应样本的起始位置和结束位置的边际信息
    """
    # calculate marginal information
    # GPU并行处理设置
    if args.n_gpu > 0:
        qa_model_marginal=torch.nn.DataParallel(qa_model_marginal)
    # 计算总评估批次大小（考虑多GPU情况）
    args.dev_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # 创建顺序采样器（保持原始数据顺序）
    train_sampler = SequentialSampler(train_dataset)
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.dev_batch_size)
    # 设置模型为评估模式
    qa_model_marginal.eval()

    # 初始化存储容器
    marginal_info_start=[]
    marginal_info_end=[]

    # 遍历数据集（显示进度条）
    for train_batch in tqdm(train_dataloader):
        # 将数据移动到指定设备
        train_batch = tuple(t.to(args.device) for t in train_batch)

        # 禁用梯度计算以节省内存
        with torch.no_grad():
            # 构建模型输入字典
            inputs = {
                "input_ids": train_batch[0],       # 输入token id
                "attention_mask": train_batch[1],  # 注意力掩码
                "token_type_ids": train_batch[2],  # 段落区分标记
            }

            # 前向传播获取模型输出
            outputs = qa_model_marginal(**inputs)
            # 对起始/结束位置logits应用sigmoid得到概率
            pred_start = torch.sigmoid(outputs[0])
            pred_end = torch.sigmoid(outputs[1])
            # 将真实标签转换为one-hot编码
            true_start = torch.nn.functional.one_hot(train_batch[4], pred_start.shape[1])
            true_end = torch.nn.functional.one_hot(train_batch[5],pred_end.shape[1])
            # 计算正确位置的概率值（点积）
            pred_start_prob = (true_start * pred_start).sum(dim=1)
            pred_end_prob = (true_end * pred_end).sum(dim=1)
            # 计算边际信息：|1 - 预测正确概率|
            # 该值越大，表示模型预测正确位置的概率越低（不确定性越高）
            diff_start = torch.abs(1 - pred_start_prob)
            diff_end = torch.abs(1 - pred_end_prob)
            # 保存当前批次的边际信息
            marginal_info_start.append(diff_start)
            marginal_info_end.append(diff_end)

    # 合并所有批次的计算结果
    marginal_info_start = torch.cat(marginal_info_start).view(-1,1)
    marginal_info_end = torch.cat(marginal_info_end).view(-1,1)
    # 合并起始和结束位置的边际信息
    marginal_info=torch.cat((marginal_info_start,marginal_info_end),dim=1)

    return marginal_info


def cal_reward_func(args, dev_dataset, qa_model, type="loss"):
    """计算问答模型在验证集上的奖励指标（支持多种评估方式）
    
    参数：
        args : 参数对象
            包含以下属性：
            - n_gpu: 可用GPU数量
            - per_gpu_eval_batch_size: 单GPU评估批大小
            - device: 计算设备（如cuda或cpu）
            - qve_eval_data_num: 采样数量（若使用随机采样）
        dev_dataset : Dataset
            用于评估的验证数据集
        qa_model : transformers模型
            需要评估的问答模型
        type : str (可选["loss", "exact", "f1"])
            评估指标类型，默认为loss
    
    返回：
        float: 缩放后的评估指标（乘以100后的百分比形式）
    """
    # 设置模型为评估模式
    qa_model.eval()

    # 计算总评估批次大小
    args.dev_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # 创建采样器（原代码中被注释的随机采样方案可用于主动学习场景）
    # dev_sampler = RandomSampler(dev_dataset,replacement=True,num_samples=args.qve_eval_data_num)
    dev_sampler = SequentialSampler(dev_dataset)

    # 创建数据加载器
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.dev_batch_size)

    total_reward = 0 # 累计奖励值

    # 遍历验证数据集
    for dev_batch in dev_dataloader:
        # 将数据移动到指定设备
        dev_batch = tuple(t.to(args.device) for t in dev_batch)

        # 禁用梯度计算
        with torch.no_grad():
            # 按类型选择计算方式
            if type == 'loss':
                # 损失模式：使用交叉熵损失作为负奖励
                inputs = {
                    "input_ids": dev_batch[0],      # 输入token id
                    "attention_mask": dev_batch[1], # 注意力掩码
                    "token_type_ids": dev_batch[2], # 段落区分标记
                    "start_positions": dev_batch[4],# 真实起始位置
                    "end_positions": dev_batch[5]   # 真实结束位置
                }

                outputs = qa_model(**inputs)
                loss = outputs[0]# 获取第一个输出（假设为损失值）

                 # 多GPU场景下取平均
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                 # 累计负损失（损失越小，奖励越高）
                total_reward -= loss*10

            elif type == 'exact':
                # 严格匹配模式：计算完全正确的样本比例
                inputs = {
                    "input_ids": dev_batch[0],  # 输入token id
                    "attention_mask": dev_batch[1],  # 注意力掩码
                    "token_type_ids": dev_batch[2],  # 段落区分标记
                }
                outputs = qa_model(**inputs)

                # 获取预测位置（取logits最大值）
                pred_start_indices = torch.argmax(outputs[0], dim=1)
                pred_end_indices = torch.argmax(outputs[1], dim=1)

                # 计算起始和结束位置都正确的样本
                start_acc = pred_start_indices == dev_batch[4]
                end_acc = pred_end_indices == dev_batch[5]
                acc_num = torch.logical_and(start_acc, end_acc).sum()

                total_reward += acc_num# 累加正确样本数
            elif type == "f1":
                 # F1模式：计算预测答案与真实答案的F1分数
                def f1_score(prediction, ground_truth):
                    #计算两个序列的F1分数
                    prediction = prediction.cpu().numpy().tolist()
                    ground_truth = ground_truth.cpu().numpy().tolist()

                    # 统计共同token数量
                    common = Counter(prediction) & Counter(ground_truth)
                    num_same = sum(common.values())

                    # 处理空预测或空真实值的情况
                    if num_same == 0:
                        return 0

                    # 计算精确度和召回率
                    precision = 1.0 * num_same / len(prediction)
                    recall = 1.0 * num_same / len(ground_truth)
                    f1 = (2 * precision * recall) / (precision + recall)
                    return f1

                inputs = {
                    "input_ids": dev_batch[0], # 输入token id
                    "attention_mask": dev_batch[1],  # 注意力掩码
                    "token_type_ids": dev_batch[2],  # 段落区分标记
                }
                outputs = qa_model(**inputs)

                # 获取预测位置（取logits最大值）
                pred_start_indices = torch.argmax(outputs[0], dim=1)
                pred_end_indices = torch.argmax(outputs[1], dim=1)

                # 逐个样本计算F1
                for i in range(len(dev_batch[0])):
                    pred_start=pred_start_indices[i]
                    pred_end = pred_end_indices[i]
                    gt_start=dev_batch[4][i]
                    gt_end = dev_batch[5][i]

                    # 提取token序列
                    prediction=dev_batch[0][i][gt_start:gt_end+1]
                    ground_truth=dev_batch[0][i][pred_start:pred_end+1]

                    total_reward+=f1_score(prediction,ground_truth)

    # 将总奖励转换为百分比形式
    return 100.0 * total_reward / len(dev_dataset)


def create_optimizer_and_scheduler(model, args, num_training_steps, learning_rate):
    """
    为模型创建优化器（AdamW）和学习率调度器（带预热的线性衰减）
    返回:
        - 如果给定总训练步数(num_training_steps), 返回优化器和调度器元组
        - 否则只返回优化器（适用于无需调度器的场景）
    """
    no_decay = ["bias", "LayerNorm.weight"] # 根据参数名称过滤特殊参数

    # 将模型参数分为两组：
    optimizer_grouped_parameters = [
        # 第一组：应用权重衰减的参数
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay)],# 筛选条件：参数名不含no_decay关键词
            "weight_decay": args.weight_decay,# 从外部参数获取衰减系数，例如0.01
        },
        # 第二组：不应用权重衰减的参数（bias/LayerNorm等）
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay)],# 参数名包含no_decay关键词
            "weight_decay": 0.0,# 显式设置为0，禁用衰减
        },
    ]

    #初始化优化器 AdamW
    optimizer = AdamW(
        optimizer_grouped_parameters,  # 使用分组后的参数
        lr=learning_rate,              # 初始学习率由外部传入（例如5e-5）
        eps=args.adam_epsilon,         # 防止除零的小量，如1e-8（从外部参数获取）
    )

    #学习率调度器配置
    if num_training_steps:
        # 当提供总训练步数时，创建线性预热+衰减调度器
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=args.warmup_steps, # 预热步数（例如500）
            num_training_steps=num_training_steps # 总训练步数（例如10000）
        )
        return optimizer,lr_scheduler  # 返回优化器和调度器的元组
    else:
        return optimizer # 无需调度器的情况（例如评估或预测任务）


def train_qa(args, model,train_dataloader, optimizer):
    """问答模型训练函数
    
    Args:
        args: 训练配置参数对象，包含以下属性：
            - n_gpu (int): 使用的GPU数量
            - fp16 (bool): 是否启用混合精度训练
            - max_grad_norm (float): 梯度裁剪的阈值
        model: 要训练的问答模型
        train_dataloader: 训练数据加载器，每次迭代返回一个批次数据
        optimizer: 优化器对象
    
    Returns:
        float: 当前epoch的平均训练损失
    """
    total_loss=0# 累计所有批次的损失值

    # 遍历训练数据的所有批次
    for step_i, batch_i in enumerate(train_dataloader):
        model.train()# 确保模型处于训练模式（影响Dropout和BatchNorm等层）

        # 准备模型输入字典
        inputs = {
            "input_ids": batch_i[0],          # 文本的token ID序列
            "attention_mask": batch_i[1],     # 注意力掩码（区分真实token和padding）
            "token_type_ids": batch_i[2],     # 分段类型ID（用于区分问题/上下文）
            "start_positions": batch_i[3],    # 答案起始位置标签
            "end_positions": batch_i[4],      # 答案结束位置标签
            "input_values": batch_i[5],       # 音频输入特征（如果是多模态模型）
        }

        # 前向传播：计算模型输出和损失
        outputs = model(**inputs)
        loss = outputs[0]# 假设模型返回的第一个元素是损失值

        # 多GPU训练时对损失求平均（当使用DataParallel时）
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        total_loss+=loss.item() # 累加损失值（转换为Python浮点数）

        # 混合精度训练处理
        if args.fp16:
            from apex import amp
            # 缩放损失并反向传播
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # 标准反向传播
            loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        if args.fp16:
            # 混合精度下的梯度裁剪需要访问优化器的主参数
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            # 标准梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()      # 更新模型参数
        model.zero_grad()     # 清空梯度缓存

    # 返回当前epoch的平均损失
    return total_loss/len(train_dataloader)


def train_qa_3p(args, model,train_dataloader, optimizer):
    """改进的问答模型训练函数，确保计算图正确管理"""
    model.train()
    total_loss = 0.0
    
    for step_i, batch_i in enumerate(train_dataloader):
        # 确保所有输入都被正确处理
        inputs = {
            "input_ids": batch_i[0].detach().requires_grad_(False),          # 不需要梯度
            "attention_mask": batch_i[1].detach().requires_grad_(False),     # 不需要梯度
            "token_type_ids": batch_i[2].detach().requires_grad_(False),     # 不需要梯度
            "start_positions": batch_i[3],                                  # 标签通常不需要梯度
            "end_positions": batch_i[4],                                    # 标签通常不需要梯度
            "input_values": batch_i[5].detach().requires_grad_(False) if batch_i[5] is not None else None,
        }
        
        # 确保梯度计算范围正确
        with torch.set_grad_enabled(True):
            outputs = model(**inputs)
            loss = outputs[0]  # 假设第一个输出是损失值
        
        if args.n_gpu > 1:
            loss = loss.mean()
            
        total_loss += loss.item()

        if args.fp16:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        # 梯度裁剪
        if args.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(train_dataloader)



def calculate_multidimensional_loss(outputs, select_prob, reward):
    # 策略梯度损失
    base_loss = -reward * torch.log(prob_values + 1e-8).mean()
    
    # 维度平衡约束（防止某个维度权重归零）
    weight_loss = torch.std(outputs['dimension_weights']) * 0.1
    
    # 各维度分数差异约束（鼓励维度区分度）
    score_cov = torch.cov(torch.stack([
        outputs['qa_score'], 
        outputs['question_score'],
        outputs['answer_score']
    ]))
    diversity_loss = -torch.mean(score_cov) * 0.05
    
    return base_loss + weight_loss + diversity_loss


def train_model(args, train_dataset, dev_dataset, train_mode, qve_model, qa_model, tokenizer):
    """联合训练QVE（问题价值评估器）和QA（问答模型）的主函数
    
    Args:
        args: 训练配置参数对象，包含以下关键属性：
            - device: 训练设备（如cuda）
            - max_steps: 最大训练步数（如果>0则优先使用）
            - gradient_accumulation_steps: 梯度累积步数
            - fp16: 是否启用混合精度训练
            - reward_type: 奖励计算类型
        train_dataset: 训练数据集
        dev_dataset: 验证数据集（用于计算奖励）
        qve_model: 问题价值评估模型
        qa_model: 问答模型
        tokenizer: 文本分词器
    
    Returns:
        无显式返回，但会保存最佳模型检查点
    """


    tb_writer = SummaryWriter()# TensorBoard日志记录器
    enhanced_output_dir = os.path.join(args.output_dir + train_mode)
    os.makedirs(enhanced_output_dir, exist_ok=True)

    # 创建日志文件路径
    log_file_path = os.path.join(enhanced_output_dir, "training_log.txt")


    ############################################
    # 训练准备阶段
    ############################################
    # 设置随机种子保证实验可重复性（需在模型初始化前设置）
    set_seed(args)

    # 计算实际训练批次大小（考虑多GPU情况）
    # 当指定max_steps时使用带重复的采样器，保证足够的训练样本
    args.train_qve_batch_size = args.per_gpu_train_qve_batch_size * max(1, args.n_gpu)
    args.train_qa_batch_size = args.per_gpu_train_qa_batch_size * max(1, args.n_gpu)

    # Data loader and number of training steps
    # 数据加载器配置
    # 当指定max_steps时使用带重复的采样器，保证足够的训练样本
    if args.max_steps > 0:
        train_sampler = RandomSampler(train_dataset, replacement=True,
                                      num_samples=args.train_qve_batch_size * args.max_steps)# 总样本数=批次大小*最大步数
    else:
        train_sampler = RandomSampler(train_dataset)
    # 创建QVE训练数据加载器
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_qve_batch_size)

    # 计算每个epoch的更新步数
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

    # 确定总训练步数和epoch数
    if args.max_steps > 0:
        t_total_qve = args.max_steps
        num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
            args.max_steps % num_update_steps_per_epoch > 0
        )
        
    else:
        t_total_qve = int(num_update_steps_per_epoch * args.num_train_epochs)
        num_train_epochs = args.num_train_epochs
        args.max_steps = t_total_qve

    # 初始化优化器和学习率调度器
    optimizer_qve, lr_scheduler_qve = create_optimizer_and_scheduler(qve_model, args, t_total_qve, args.qve_learning_rate)
    optimizer_qa = create_optimizer_and_scheduler(qa_model, args, None, args.learning_rate)

    current_temperature = 1.0  # 初始温度
    target_std = 0.2  # 目标标准差 (0.15-0.25)
    
    # 熵正则化系数
    entropy_coeff = 0.1
    
    # 探索系数 (随时间衰减)
    exploration_coeff = 1.0
    exploration_decay = 0.95  # 每1000步衰减一次

    #稳定性参数
    stability_eps = 1e-7

    # 混合精度训练初始化
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        qve_model, optimizer_qve = amp.initialize(qve_model, optimizer_qve, opt_level=args.fp16_opt_level)
        qa_model, optimizer_qa = amp.initialize(qa_model, optimizer_qa, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    # 多GPU并行训练
    if args.n_gpu > 1:
        qve_model = torch.nn.DataParallel(qve_model)
        qa_model = torch.nn.DataParallel(qa_model)

    total_train_batch_size_qve = (
            args.train_qve_batch_size
            * args.gradient_accumulation_steps
    )
    # 日志输出训练配置信息
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per device QVE = %d, QA = %d", args.per_gpu_train_qve_batch_size, args.per_gpu_train_qa_batch_size,)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) QVE = %d, QA = %d", total_train_batch_size_qve, args.train_qa_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps QVE = %d", t_total_qve)

    # 初始化训练跟踪变量
    global_step = 0 # 全局训练步数计数器
    epochs_trained = 0 # 已完成的epoch数

    # 累积损失和指标（使用设备张量减少数据传输）
    tr_loss_qa = torch.tensor(0.0).to(args.device)    # QA模型损失累积
    tr_loss_qve = torch.tensor(0.0).to(args.device)  # QVE模型损失累积
    tr_reward = torch.tensor(0.0).to(args.device)    # 奖励信号累积
    tr_qa_acc = torch.tensor(0.0).to(args.device)    # QA准确率累积

    # 最佳模型跟踪指标
    best_reward = -100000  # 最佳奖励跟踪
    lowest_loss = 100000   # 最低损失跟踪

    # 滑动平均计算的中间变量
    logging_qa_loss_scalar,logging_qve_loss_scalar,logging_reward_scalar,logging_qa_acc_scalar  = 0.0, 0.0, 0.0, 0.0
    # 计算基线性能（初始QA模型在验证集的表现）
    baseline_performance=cal_reward_func(args, dev_dataset, qa_model, type=args.reward_type)

    # 初始化模型梯度
    qa_model.zero_grad()
    qve_model.zero_grad()

    # 初始化训练进度条（epoch级别）
    train_pbar = trange(epochs_trained, int(np.ceil(num_train_epochs)), desc="Epoch")

    # 保存QA模型的初始状态（用于后续重置）
    qa_model_init_statedict = copy.deepcopy(qa_model).state_dict()

    entropy = None
    mean_entropy = None

    ############################################
    # 开始训练循环
    ############################################
    # 开始epoch循环
    for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):
        epoch_iterator = train_dataloader# 获取当前epoch的数据迭代器
        epoch_pbar = tqdm(epoch_iterator, desc="Iteration") # 批次级别的进度条

        # 遍历每个batch
        for step, batch in enumerate(epoch_iterator):
            ############################################
            # QVE筛选阶段
            ############################################
            

            batch = tuple(t.to(args.device) for t in batch)# 将batch数据移动到指定设备（如GPU）
            
            with torch.no_grad():
                # 生成伪标签（基于QA预测的可回答性概率）
                # batch[9]的形状为(batch_size, num_questions)，每个元素是问题可回答性概率
                # 计算每个样本的平均可回答性，并取前selected_question_percentage%作为正样本
                threshold1 = (1-batch[12]).mean(dim=1).sort()[0][int(len(batch[12])*(1-args.selected_question_percentage))]
                noisy_GT = (1-batch[12]).mean(dim=1) >= threshold1# 生成噪声标签

                #选择对应模型的掩码
                attention_mask_qve = batch[7]
                if (train_mode == "qa"):
                    attention_mask_qve = batch[8]
                if (train_mode == "ca"):
                    attention_mask_qve = batch[9]
                if (train_mode == "qc"):
                    attention_mask_qve = batch[10]

            qve_model.train()# 确保QVE模型处于训练模式
            # 准备QVE模型输入
            inputs = {
                "input_ids": batch[6],
                "attention_mask": attention_mask_qve,
                "token_type_ids": batch[11],
                "marginal_info": batch[12],
            }

            # ===================================================================
            # 核心改进1: 动态温度缩放
            # ===================================================================
            raw_logits = qve_model(**inputs)[0]

            current_temperature = max(current_temperature, 0.01)

            scaled_logits = raw_logits / current_temperature

            if scaled_logits.size(1) == 1:
                qa_values = torch.sigmoid(scaled_logits).view(-1)
            else:
                # 使用更稳定的softmax计算
                stable_logits = scaled_logits - scaled_logits.max(dim=1, keepdim=True).values
                exp_logits = torch.exp(stable_logits)
                qa_values = exp_logits[:, 1] / (exp_logits.sum(dim=1) + stability_eps)
                qa_values = qa_values.view(-1)
                
            # 避免数值不稳定
            qa_values = torch.clamp(qa_values, min=stability_eps, max=1.0 - stability_eps)

            # 计算熵并保存
            entropy = - (qa_values * torch.log(qa_values) + (1 - qa_values) * torch.log(1 - qa_values))
            mean_entropy = entropy.mean()
            current_std = qa_values.std().item()

            # 3. 采样 (没有梯度)
            with torch.no_grad():
                threshold2 = qa_values.sort()[0][int(len(batch[9]) * (1 - args.selected_question_percentage))]
                estimated_label = qa_values >= threshold2
                noise_acc_qve = 1.0 * (noisy_GT == estimated_label).sum() / len(noisy_GT)
                
                # Gumbel-Softmax采样（增加稳定性）
                uniform_noise = torch.rand_like(qa_values)
                gumbel_noise = -torch.log(-torch.log(uniform_noise + stability_eps) + stability_eps)
                
                # 检查并修复非数值
                if torch.isnan(gumbel_noise).any():
                    gumbel_noise = torch.zeros_like(qa_values)
                    
                gumbel_logits = torch.log(qa_values/(1 - qa_values + stability_eps)) + gumbel_noise
                select_prob = torch.sigmoid(gumbel_logits)
                
                # 确保采样概率有效
                select_prob = torch.clamp(select_prob, 0.0, 1.0)

            # 4. QA模型训练
            qa_inputs = TensorDataset(
                batch[0], batch[1], batch[2], batch[4], batch[5], select_prob
            )
            train_QA_sampler = RandomSampler(qa_inputs)
            train_QA_dataloader = DataLoader(qa_inputs, sampler=train_QA_sampler, 
                                           batch_size=args.train_qa_batch_size)
            
            qa_loss = train_qa(args, qa_model, train_QA_dataloader, optimizer_qa)
            tr_loss_qa += qa_loss

            # 5. 奖励计算 (没有梯度)
            with torch.no_grad():
                cur_qa_performance = cal_reward_func(args, dev_dataset, qa_model, type=args.reward_type)
                reward = cur_qa_performance - baseline_performance
                tr_qa_acc += cur_qa_performance
                tr_reward += reward
            
            log_qa = torch.log(qa_values + stability_eps)
            log_not_qa = torch.log(1 - qa_values + stability_eps)
            
            # 2. 奖励缩放（防止过大的奖励值影响稳定性）
            scaled_reward = reward * 0.01  # 缩放因子
            
            # 3. 安全的概率计算
            prob = select_prob * log_qa + (1 - select_prob) * log_not_qa
            
            # 4. 检查概率值并防止NaN
            if torch.isnan(prob).any():
                prob = torch.zeros_like(prob) + stability_eps  # 重置为小正值
                
            # 5. 计算损失（使用缩放后的奖励）
            qve_loss_rl = -scaled_reward * prob.mean()
            
            # 探索辅助损失
            threshold = 0.5 + 0.3 * (epoch / num_train_epochs)
            qve_loss_aux = torch.relu(qa_values.mean() - threshold) + torch.relu(1 - threshold - qa_values.mean())
            
            
            qve_loss_total = qve_loss_rl + qve_loss_aux - entropy_coeff * mean_entropy
            variance_loss = torch.relu(target_std - qa_values.std()) * 0.1
            qve_loss_total += variance_loss
            
            # 6. QVE反向传播
            if args.gradient_accumulation_steps > 1:
                qve_loss_total = qve_loss_total / args.gradient_accumulation_steps
                
            tr_loss_qve += qve_loss_total
            
            if args.fp16:
                with amp.scale_loss(qve_loss_total, optimizer_qve) as scaled_loss:
                    scaled_loss.backward()
            else:
                qve_loss_total.backward()
            
            # 7. QVE梯度更新
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_qve), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(qve_model.parameters(), args.max_grad_norm)
                
                optimizer_qve.step()
                lr_scheduler_qve.step()
                qve_model.zero_grad()
                
                global_step += 1
                
                # 动态参数调整
                if global_step % 10 == 0:
                    if not torch.isnan(qa_values).any():
                        current_std = qa_values.std().item()
                        if current_std < target_std - 0.05:
                            current_temperature = max(current_temperature * 0.9, 0.01)  # 限制下限
                        elif current_std > target_std + 0.05:
                            current_temperature = min(current_temperature * 1.1, 10.0)  # 限制上限
                    else:
                        # 重置温度到安全值
                        current_temperature = 1.0
                
                if global_step % 1000 == 0:
                    exploration_coeff = max(0.1, exploration_coeff * exploration_decay)
    

                 # 定期日志记录（按logging_steps配置）
                if (args.logging_steps > 0 and global_step % args.logging_steps == 0):
                    logs = {}

                    def safe_stat(x):
                        return x.item() if not torch.isnan(x).any() else 0.0

                     # 计算各指标的标量值
                    tr_loss_qa_scalar = tr_loss_qa.item()
                    tr_loss_qve_scalar=tr_loss_qve.item()
                    tr_qa_acc_scalar=tr_qa_acc.item()
                    tr_reward_scalar=tr_reward.item()
                    logs["qa_loss"] = (tr_loss_qa_scalar - logging_qa_loss_scalar) / args.logging_steps
                    logs["eval_qa_current"] = (tr_qa_acc_scalar - logging_qa_acc_scalar) / args.logging_steps
                    logs["reward"] = (tr_reward_scalar - logging_reward_scalar) / args.logging_steps
                    logs["qve_loss_total"] = (tr_loss_qve_scalar - logging_qve_loss_scalar) / args.logging_steps
                    logs["noise_acc_qve"] = noise_acc_qve.item() 
                    logs['eval_qa_baseline'] = baseline_performance.item()
                    
                    # 安全获取选择问题数量
                    if not torch.isnan(select_prob).any():
                        logs['num of selected questions'] = select_prob.sum().item()
                    else:
                        logs['num of selected questions'] = 0
                        
                    # 安全获取其他统计数据
                    if not torch.isnan(qa_values).any():
                        logs["qa_values_std"] = qa_values.std().item()
                        logs["min_prob"] = qa_values.min().item()
                        logs["max_prob"] = qa_values.max().item()
                        logs["mean_prob"] = qa_values.mean().item()
                        logs["median_prob"] = qa_values.median().item()
                        logs["entropy"] = mean_entropy.item()
                    else:
                        # 使用占位值
                        logs["qa_values_std"] = 0.0
                        logs["min_prob"] = 0.0
                        logs["max_prob"] = 0.0
                        logs["mean_prob"] = 0.0
                        logs["median_prob"] = 0.0
                        logs["entropy"] = 0.0
                        
                    logs["temperature"] = current_temperature
                    logs["exploration"] = exploration_coeff

                    # 更新滑动平均基准值
                    logging_qa_loss_scalar = tr_loss_qa_scalar
                    logging_qve_loss_scalar = tr_loss_qve_scalar
                    logging_reward_scalar = tr_reward_scalar
                    logging_qa_acc_scalar = tr_qa_acc_scalar

                    # 输出日志到控制台和TensorBoard
                    logger.info(logs)

                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"Step {global_step} - ")
                        for key, value in logs.items():
                            log_file.write(f"{key}: {value:.4f} ")
                        log_file.write("\n")

                    # 保存最佳模型逻辑
                    save_flag = False
                    # 按奖励保存最佳模型
                    if logs["reward"] > best_reward:
                        best_reward = logs["reward"]
                        output_dir = os.path.join(enhanced_output_dir, "checkpoint-best-reward")
                        save_flag = True
                    # 按损失保存最佳模型
                    if logs["qa_loss"] < lowest_loss:
                        lowest_loss = logs["qa_loss"]
                        output_dir = os.path.join(enhanced_output_dir, "checkpoint-best-loss")
                        save_flag = True

                    # 执行模型保存
                    if save_flag:
                        # Take care of distributed/parallel training
                        # 处理多GPU情况
                        model_to_save = qve_model.module if hasattr(qve_model, "module") else qve_model
                        # 保存模型、分词器和训练参数
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Step: %d, Saving qve_model checkpoint to %s", global_step, output_dir)

                # 重置QA模型到初始状态（准备下一个QVE筛选阶段）
                qa_model.load_state_dict(copy.deepcopy(qa_model_init_statedict))

            # 定期保存检查点（按save_steps配置）
            if global_step % args.save_steps == 0:
                output_dir = os.path.join(enhanced_output_dir, "checkpoint-{}".format(global_step))
                # Take care of distributed/parallel training
                model_to_save = qve_model.module if hasattr(qve_model,"module") else qve_model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving qve_model checkpoint to %s", output_dir)
                # 保存优化器和调度器状态（用于恢复训练）
                torch.save(optimizer_qve.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(lr_scheduler_qve.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

            # 更新批次进度条
            epoch_pbar.update(1)
            # 提前终止（达到最大步数）
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        epoch_pbar.close()
        train_pbar.update(1)

        # 提前终止（达到最大步数）
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    ############################################
    # 训练收尾工作
    ############################################
    train_pbar.close()
    if tb_writer:
        tb_writer.close()# 关闭TensorBoard写入器

    logger.info("\n\nTraining completed.\n\n")

    return

def train_weights(args, train_dataset, dev_dataset, qve_models, qa_model, optimizer_weights, tokenizer, dynamic_weight_model):
    """联合训练QVE（问题价值评估器）和QA（问答模型）的主函数
    
    Args:
        args: 训练配置参数对象，包含以下关键属性：
            - device: 训练设备（如cuda）
            - max_steps: 最大训练步数（如果>0则优先使用）
            - gradient_accumulation_steps: 梯度累积步数
            - fp16: 是否启用混合精度训练
            - reward_type: 奖励计算类型
        train_dataset: 训练数据集
        dev_dataset: 验证数据集（用于计算奖励）
        qve_model: 问题价值评估模型
        qa_model: 问答模型
        tokenizer: 文本分词器
    
    Returns:
        无显式返回，但会保存最佳模型检查点
    """
    ############################################
    # 训练准备阶段
    ############################################
    # 设置随机种子保证实验可重复性（需在模型初始化前设置）
    set_seed(args)

    # 计算实际训练批次大小（考虑多GPU情况）
    # 当指定max_steps时使用带重复的采样器，保证足够的训练样本？
    args.train_qve_batch_size = 16
    args.train_qa_batch_size = 64  # 增大QA批大小加速训练

    # Data loader and number of training steps
    # 数据加载器配置
    # 当指定max_steps时使用带重复的采样器，保证足够的训练样本
    if args.max_steps > 0:
        train_sampler = RandomSampler(train_dataset, replacement=True,
                                      num_samples=args.train_qve_batch_size * args.max_steps)# 总样本数=批次大小*最大步数
    else:
        train_sampler = RandomSampler(train_dataset)

    epoch_threshold1 = int(args.weight_epochs * 0.4)  # 40%轮数为早期
    epoch_threshold2 = int(args.weight_epochs * 0.7)  # 70%轮数为中期


    # 奖励缓存和滑动平均参数
    reward_history = []
    reward_ema = 0.0  # 指数移动平均值
    EMA_FACTOR = 0.9   # EMA平滑因子
    EVAL_STEPS = {
        "early": 20,   # 早期评估频率
        "mid": 10,     # 中期评估频率
        "late": 5      # 后期评估频率
    }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_qve_batch_size * 8,  # 增大批次加速训练
        sampler=RandomSampler(train_dataset, num_samples=40000),  # 减少样本数
        drop_last=True
    )

    CUDA_VISIBLE_DEVICES=1

    optimizer_qa = create_optimizer_and_scheduler(qa_model, args, None, args.learning_rate)
    qa_model_init_state_dict = copy.deepcopy(qa_model).state_dict()

    # 创建权重保存目录
    weights_dir = os.path.join(args.output_dir, "qve_weights")
    os.makedirs(weights_dir, exist_ok=True)

    # 日志输出训练配置信息
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.weight_epochs)
    # 训练循环
    logger.info(f"总轮数: {args.weight_epochs} | 批次大小: {train_dataloader.batch_size}")
    for epoch in range(args.weight_epochs):
        if epoch < epoch_threshold1:
            eval_frequency = EVAL_STEPS["early"]
            phase = "早期"
        elif epoch < epoch_threshold2:
            eval_frequency = EVAL_STEPS["mid"]
            phase = "中期"
        else:
            eval_frequency = EVAL_STEPS["late"]
            phase = "后期"

        logger.info(f"== 轮次 {epoch+1}/{args.weight_epochs} ({phase}) | 评估频率: 每{eval_frequency}步 ===")

        epoch_iterator = train_dataloader
        dynamic_weight_model.train()
        epoch_rewards = []
        reward_baseline = cal_reward_func(args, dev_dataset, qa_model, type=args.reward_type)
        last_full_eval = 0  # 记录上次完整评估的step

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            
            ############################################
            # 1. 获取三个模型的分数
            ############################################
            model_scores = []
            with torch.no_grad():
                for mode in ["ca", "qc", "qa"]:
                    # 构建输入 - 根据您的实际数据结构调整

                    attention_mask_qve = batch[7]
                    if (mode == "qa"):
                        attention_mask_qve = batch[8]
                    if (mode == "ca"):
                        attention_mask_qve = batch[9]
                    if (mode == "qc"):
                        attention_mask_qve = batch[10]

                    # 准备QVE模型输入
                    inputs = {
                        "input_ids": batch[6],        # 问题/上下文的token IDs
                        "attention_mask": attention_mask_qve,   # 注意力掩码
                        "token_type_ids": batch[11],    # 段落分隔标记（如BERT需要）
                        "marginal_info": batch[12],    # 边际信息（可回答性概率）
                    }
                    
                    qa_values = qve_models[mode](**inputs)[0]

                    if qa_values.size(1) == 1:# 二分类（sigmoid输出）
                        qa_values = qa_values.view(-1)
                    else:# 多分类（softmax后取正类概率）
                        qa_values = torch.softmax(qa_values, dim=1)[:, 1].view(-1)
                    
                    model_scores.append(qa_values)
            
            # 堆叠分数 [batch_size, num_models]
            raw_scores = torch.stack(model_scores, dim=-1)
            
            ############################################
            # 2. 动态权重计算最终分数
            ############################################
            combined_scores = dynamic_weight_model(raw_scores)
            qa_values = torch.sigmoid(combined_scores)  # 应用sigmoid得到概率
            
            ############################################
            # 3. 采样、QA训练和奖励计算
            ############################################
            select_prob = Binomial(1, qa_values).sample()
            
            # QA训练 
            inputs = TensorDataset(batch[0], batch[1], batch[2], batch[4], batch[5], select_prob)
            train_QA_sampler = RandomSampler(inputs)
            train_QA_dataloader = DataLoader(inputs, sampler=train_QA_sampler, batch_size=args.train_qa_batch_size)

            # 调用QA训练函数（返回损失值）
            qa_model.load_state_dict(qa_model_init_state_dict)  # 重置
            qa_loss = train_qa(args, qa_model, train_QA_dataloader, optimizer_qa)
            
            ############################################
            # 4. 奖励计算（使用原函数方法）
            ############################################
            do_full_eval = (step % eval_frequency == 0) or (step == 0)

            if do_full_eval:
                # 完整评估
                cur_performance = cal_reward_func(args, dev_dataset, qa_model, type=args.reward_type)
                reward_diff = cur_performance - reward_baseline
                
                # 自适应放大奖励
                min_diff = 0.005  # 更敏感阈值
                if abs(reward_diff) > min_diff:
                    amplify_factor = min(50, max(2, int(0.5 / abs(reward_diff))))
                    reward = reward_diff * amplify_factor
                else:
                    reward = 0.0
                
                # 更新EMA和基线
                reward_ema = EMA_FACTOR * reward_ema + (1 - EMA_FACTOR) * reward
                reward_history.append(reward)
                reward_baseline = reward_baseline * 0.9 + cur_performance * 0.1
                last_full_eval = step
                
                logger.debug(f"执行完整评估 | 奖励: {reward:.4f} | 放大系数: {amplify_factor}")
            else:
                # 使用EMA近似值
                reward = reward_ema
                # 小概率随机评估防止偏离
                if random.random() < 0.05:  # 5%概率额外评估
                    cur_performance = cal_reward_func(args, dev_dataset, qa_model, type=args.reward_type)
                    reward_diff = cur_performance - reward_baseline
                    if abs(reward_diff) > 0.5:  # 发生显著偏离
                        reward = reward_diff * min(20, int(1/abs(reward_diff)))
                        reward_ema = EMA_FACTOR * reward_ema + (1 - EMA_FACTOR) * reward
                        logger.warning(f"随机校验发现偏离 | 新奖励: {reward:.4f}")

            # 更新基线
            epoch_rewards.append(reward)

            ############################################
            # 5. 权重更新（策略梯度）
            ############################################
            log_prob = select_prob * torch.log(qa_values + 1e-8) + (1 - select_prob) * torch.log(1 - qa_values + 1e-8)
            weight_loss = -reward * log_prob.mean()
            
            # 权重多样性正则化
            weights = dynamic_weight_model.normalized_weights
            entropy = -torch.sum(weights * torch.log(weights + 1e-8))
            weight_loss -= args.entropy_coef * entropy
            
            # 优化器更新
            optimizer_weights.zero_grad()
            weight_loss.backward()
            torch.nn.utils.clip_grad_norm_(dynamic_weight_model.parameters(), 1.0)
            optimizer_weights.step()

            
            ############################################
            # 6. 日志记录和权重历史
            ############################################
            if step % 10 == 0:
                # 简化日志频率
                weights_softmax = dynamic_weight_model.normalized_weights.cpu().numpy()
                logger.info(
                    f"E{epoch} S{step} | W: {weights_softmax.round(3)} "
                    f"T: {dynamic_weight_model.temperature.item():.3f} | "
                    f"Loss: {weight_loss.item():.4f} | R: {reward:.4f} "
                    f"({'F' if do_full_eval else 'A'})"  # F表示完整评估，A表示近似
                )
                
        # ===== 轮次结束处理 =====
        # 保存检查点（轮次结束时强制验证）
        final_reward = cal_reward_func(args, dev_dataset, qa_model, type=args.reward_type)
        save_path = os.path.join(args.output_dir, f"weights_epoch_{epoch}.pt")
        dynamic_weight_model.save(save_path)
        logger.info(f"✅ 保存第{epoch}轮权重 | 最终奖励: {final_reward:.4f}")
        
        # 后期增强学习率衰减
        if epoch >= epoch_threshold2:
            for param_group in optimizer_weights.param_groups:
                param_group['lr'] *= 0.85
            logger.info(f"降低学习率至: {param_group['lr']:.6f}")

    return dynamic_weight_model

def load_and_cache_examples(args, tokenizer, qa_model_marginal=None, output_examples=False, dev=False):
    """
    加载并缓存预处理后的数据集，支持训练集/验证集的切换
    
    参数说明：
    - args: 包含运行参数的配置对象（如最大序列长度、步幅等）
    - tokenizer: 用于文本分词的tokenizer实例
    - qa_model_marginal: 用于计算边缘概率的QA模型（可选，用于训练阶段）
    - output_examples: 是否返回原始示例数据（用于调试或分析）
    - dev: 是否为验证集模式（True则加载验证集，False加载训练集）
    
    返回：
    - 预处理后的PyTorch数据集（TensorDataset）
    - 当output_examples=True时，额外返回原始示例和特征数据
    """
    # 确定数据集模式（训练/验证）
    mode="dev" if dev else "train"

    # 解析QA数据集名称
    qa_dataset_name = [n for n in ["NewsQA","NaturalQuestionsShort","HotpotQA","TriviaQA-web"] if n in args.train_file][0]
    # 构建缓存文件路径，格式示例：cached_train_HotpotQA_512
    cached_features_file = os.path.join(
        "cached_{}_{}_{}".format(
            mode,
            qa_dataset_name,
            str(args.max_seq_length),
        ),
    )

    # 缓存加载逻辑（如果存在且不需要覆盖）
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        # 从缓存加载预处理结果
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        # 确定要处理的原始数据文件路径
        process_file = args.dev_file if dev else args.train_file
        logger.info("Creating features from dataset file at: %s", process_file)

        # 初始化SQuAD数据处理工具（适用于类似SQuAD格式的QA数据集）
        processor = SquadV1Processor()
        # 获取原始示例数据
        examples = processor.get_train_examples(data_dir=None, filename=process_file)

        # 检查前3个样本
        # for example in examples[:3]:  
        #     logger.info(f"问题ID: {example.qas_id}")
        #     logger.info(f"问题: {example.question_text}")
        #     logger.info(f"答案（新context）: {example.context}\n{'-'*50}")   

        # 打印前3个示例的调试信息（验证数据加载正确性）
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

        # 当存在qa_model_marginal且为训练模式时，计算边缘概率信息
        if qa_model_marginal and not dev:
            # 计算每个样本的边际概率（用于多任务学习或辅助训练）
            marginal_info = calculate_marginal_info(qa_model_marginal, dataset, args)
            # 将边际概率作为新维度添加到数据集
            new_data = dataset.tensors + (marginal_info.cpu(),)
            dataset = TensorDataset(*new_data)

        # 保存预处理结果到缓存文件
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    # 打印第一个样本的输入序列示例（用于调试分词结果）
    sample_input = tokenizer.decode(features[0].input_ids)
    logger.info(f"qve输入序列示例：\n{sample_input}")

    first_sample_input_ids = dataset.tensors[0][0].tolist()
    # 使用 tokenizer 解码这些 input_ids
    first_sample_text = tokenizer.decode(first_sample_input_ids, skip_special_tokens=True)
    # 打印解码后的文本内容
    #print(f"qve的dataset第一个样本的输入序列示例：\n{first_sample_text}")    


    # 根据参数决定返回内容
    if output_examples:
        return dataset, examples, features
    return dataset


def load_and_cache_3p_examples(args, tokenizer, qa_model_marginal=None, output_examples=False, dev=False):
    """
    加载并缓存预处理后的数据集，支持多种输入模式
    
    参数说明：
    - args: 包含运行参数的配置对象
    - tokenizer: 用于文本分词的tokenizer实例
    - mode: 输入模式 ('qa', 'q_c', 'q_a', 'c_a')
    - qa_model_marginal: 用于计算边缘概率的QA模型（可选）
    - output_examples: 是否返回原始示例数据
    - dev: 是否为验证集模式
    
    返回：
    - 预处理后的PyTorch数据集
    - 当output_examples=True时，额外返回原始示例和特征数据
    """
    # 确定数据集模式（训练/验证）
    dataset_mode = "dev" if dev else "train"

    # 解析QA数据集名称
    qa_dataset_name = [n for n in ["NewsQA","NaturalQuestionsShort","HotpotQA","TriviaQA-web"] 
                      if n in args.train_file][0]
    
    # 修改缓存文件名以包含输入模式
    cached_features_file = os.path.join(
        "cached_{}_{}_{}_3p".format(
            dataset_mode,
            qa_dataset_name,
            str(args.max_seq_length),
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        # 从缓存加载预处理结果
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        # 确定要处理的原始数据文件路径
        process_file = args.dev_file if dev else args.train_file
        logger.info("Creating features from dataset file at: %s", process_file)

        # 初始化SQuAD数据处理工具（适用于类似SQuAD格式的QA数据集）
        processor = SquadV1Processor()
        # 获取原始示例数据
        examples = processor.get_train_examples(data_dir=None, filename=process_file)

        # # 检查前3个样本
        # for example in examples[:3]:  
        #     logger.info(f"问题ID: {example.qas_id}")
        #     logger.info(f"问题: {example.question_text}")
        #     logger.info(f"答案（新context）: {example.context}\n{'-'*50}")   

        # 打印前3个示例的调试信息（验证数据加载正确性）
        features, dataset = squad_convert_3p_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True,
            return_dataset="pt",
            threads=args.threads,
        )

        # 当存在qa_model_marginal且为训练模式时，计算边缘概率信息
        if qa_model_marginal and not dev:
            # 计算每个样本的边际概率（用于多任务学习或辅助训练）
            marginal_info = calculate_marginal_info(qa_model_marginal, dataset, args)
            # 将边际概率作为新维度添加到数据集
            new_data = dataset.tensors + (marginal_info.cpu(),)
            dataset = TensorDataset(*new_data)

        # 保存预处理结果到缓存文件
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    # 打印第一个样本的输入序列示例（用于调试分词结果）
    sample_input = tokenizer.decode(features[0].input_ids)
    #logger.info(f"qve输入序列示例：\n{sample_input}")

    first_sample_input_ids = dataset.tensors[0][0].tolist()
    # 使用 tokenizer 解码这些 input_ids
    first_sample_text = tokenizer.decode(first_sample_input_ids, skip_special_tokens=True)
    # 打印解码后的文本内容
    #logger.info(f"qve的dataset第一个样本的输入序列示例：\n{first_sample_text}")    

    # 根据参数决定返回内容
    if output_examples:
        return dataset, examples, features
    return dataset


def estimation(args, tokenizer, qve_model, input_mode, qa_model):
    # 加载数据集并缓存特征（特征包含分词后的输入信息）
    # 返回的dataset是模型可处理的特征集合，examples是原始样本，features是预处理后的特征
    dataset, examples, features = load_and_cache_3p_examples(args, tokenizer, qa_model, output_examples=True, dev=False)

    # 多GPU并行处理
    if args.n_gpu > 0:
        qve_model=torch.nn.DataParallel(qve_model)
    qve_model.eval()# 设置为评估模式

    # 设置评估批大小（根据GPU数量调整）
    args.dev_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    data_sampler = SequentialSampler(dataset)# 顺序采样器（不随机）
    dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=args.dev_batch_size)

    # 日志输出基本信息(进行评估)
    logger.info("***** Running Question Value Estimation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.dev_batch_size)

    # 初始化存储结果的容器
    all_qa_values = [] # 存储所有问题的价值评估得分
    all_qa_answerabilities = [] # 存储所有问题的可回答性得分

    # 逐批次处理数据
    for batch in tqdm(dataloader, desc="Estimation"):
        batch = tuple(t.to(args.device) for t in batch) # 将数据移动到指定设备（如GPU）
        with torch.no_grad():# 禁用梯度计算以节省内存
            # 构建模型输入
            attention_mask_qve = batch[7]
            if (input_mode == "qa"):
                attention_mask_qve = batch[8]
            if (input_mode == "ca"):
                attention_mask_qve = batch[9]
            if (input_mode == "qc"):
                attention_mask_qve = batch[10]

            # 准备QVE模型输入
            inputs = {
                "input_ids": batch[6],        # 问题/上下文的token IDs
                "attention_mask": attention_mask_qve,   # 注意力掩码
                "token_type_ids": batch[11],    # 段落分隔标记（如BERT需要）
                "marginal_info": batch[12],    # 边际信息（可回答性概率）
            }


            # 模型直接输出回归值，无需条件判断
            qa_values = qve_model(**inputs)[0].view(-1)

            # 对回归结果应用sigmoid归一化到[0,1]
            qa_values_normalized = torch.sigmoid(qa_values)

            all_qa_values.append(qa_values_normalized)
            all_qa_answerabilities.append((1 - batch[12]).mean(dim=1))

    # 合并所有批次的评估结果
    all_qa_values = torch.cat(all_qa_values).cpu().detach().numpy()
    all_qa_answerabilities = torch.cat(all_qa_answerabilities)

    # # 将结果转为numpy数组（后续处理可能需要）
    # # all_qa_values = Binomial(1, all_qa_values).sample()
    # all_qa_values = all_qa_values.cpu().detach().numpy()

    # 聚合每个问题的最大价值得分
    qid2feature = {}
    for ii, feature in enumerate(features):
        if feature.qas_id not in qid2feature:
            qid2feature[feature.qas_id] = [ii]
        else:
            qid2feature[feature.qas_id].append(ii)

    # 聚合每个问题的最大价值得分
    qid2qv = {}
    for qid, fids in qid2feature.items():   
        qid2qv[qid] = max(all_qa_values[fids])# 取同一问题的多个特征中的最高分

    # 计算筛选前的平均分（0-100）
    pre_filter_scores = list(qid2qv.values())
    pre_avg_score = (sum(pre_filter_scores) / len(pre_filter_scores)) * 100 if pre_filter_scores else 0.0

    # 按分数排序问题ID，并截取前N%
    sorted_qid_scores = sorted(qid2qv.items(), key=lambda x: x[1], reverse=True)
    selected_num = int(len(sorted_qid_scores) * args.selected_question_percentage)
    filtered_id_list = [qid for qid, _ in sorted_qid_scores[:selected_num]]

    # 计算筛选后的平均分（0-100）
    post_filter_scores = [score for qid, score in sorted_qid_scores[:selected_num]]
    post_avg_score = (sum(post_filter_scores) / len(post_filter_scores)) * 100 if post_filter_scores else 0.0

    # 计算分数阈值（入选的最低分，0-100）
    score_threshold = (sorted_qid_scores[selected_num-1][1] * 100) if selected_num > 0 else 0.0

    # 输出统计信息
    logger.info(f"筛选前平均分: {pre_avg_score:.2f} (范围 0-100)")
    logger.info(f"筛选后平均分: {post_avg_score:.2f} (范围 0-100)")
    logger.info(f"筛选阈值: {score_threshold:.2f} (分数≥此值的问题被选中)")


    # 读取原始训练文件进行过滤
    # filtered_id_list = [qid for qid,qv in qid2qv.items() if qv==1]
    ##write to json
    data_json = json.load(open(args.train_file, 'r'))
    new_passages_train = []

    # 逐层重建过滤后的数据结构
    for passages in data_json['data']:
        new_paras_train = []

        for para in passages['paragraphs']:
            context = para['context']
            new_qas_train = []

            for qa in para['qas']:
                if qa['id'] in filtered_id_list:# 保留被选中的问题
                    new_qas_train.append(qa)

            if len(new_qas_train) > 0:
                new_paras_train.append({'context': context, 'qas': new_qas_train})

        if len(new_paras_train) > 0:
            new_passages_train.append({'title': passages['title'], 'paragraphs': new_paras_train})

    # 构建最终过滤后的数据对象
    filtered_data_json = {'data': new_passages_train, 'version': data_json['version']}
 
    # 统计过滤前后的数据量对比
    total = 0
    context_num = 0
    for paras in data_json['data']:
        for para in paras['paragraphs']:
            context_num += 1
            qa_num = len(para['qas'])
            total += qa_num
    logger.info('Before filtering: Train QA Num: %d, Total Context: %d' % (total, context_num))

 
    total = 0
    context_num = 0
    for paras in filtered_data_json['data']:
        for para in paras['paragraphs']:
            context_num += 1
            qa_num = len(para['qas'])
            total += qa_num
    logger.info('After filtering: Train QA Num: %d, Total Context: %d' % (total, context_num))

    # 保存过滤后的数据集
    filename = f"filtered_score_" + input_mode + ".json"

    logger.info("Saving results to %s", filename)
    enhanced_output_dir = os.path.join(args.output_dir)
    json.dump(filtered_data_json, open(os.path.join(enhanced_output_dir, filename), 'w'))

    return


def estimation_3p(args, tokenizer, qve_models, qa_model,dynamic_weight_model, loss_weight):
    # 加载数据集并缓存特征（特征包含分词后的输入信息）
    # 返回的dataset是模型可处理的特征集合，examples是原始样本，features是预处理后的特征
    dataset, examples, features = load_and_cache_3p_examples(args, tokenizer, qa_model, output_examples=True, dev=False)

    # 多GPU并行处理
    if args.n_gpu > 0:
        for input_mode in ["ca","qc", "qa"]:
            qve_models[input_mode] = torch.nn.DataParallel(qve_models[input_mode])
    for input_mode in ["ca","qc", "qa"]:
        qve_models[input_mode].eval()# 设置为评估模式

    # 设置评估批大小（根据GPU数量调整）
    args.dev_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    data_sampler = SequentialSampler(dataset)# 顺序采样器（不随机）
    dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=args.dev_batch_size)

    # 日志输出基本信息(进行评估)
    logger.info("***** Running Question Value Estimation *****")
    if ( loss_weight!=""):
        logger.info("***** loss_weight:%s *****", loss_weight)
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.dev_batch_size)

    # 初始化存储结果的容器
    all_qa_values = [] # 存储所有问题的价值评估得分
    all_qa_answerabilities = [] # 存储所有问题的可回答性得分

    # 逐批次处理数据
    for batch in tqdm(dataloader, desc="Estimation"):
        batch = tuple(t.to(args.device) for t in batch) # 将数据移动到指定设备（如GPU）
        with torch.no_grad():# 禁用梯度计算以节省内存
            # 构建模型输入
            model_scores = []
            for input_mode in ["ca","qc", "qa"]:
                # qa_values = 0.0
                # if (loss_weight != input_mode):
                attention_mask_qve = batch[7]
                if (input_mode == "qa"):
                    attention_mask_qve = batch[8]
                if (input_mode == "ca"):
                    attention_mask_qve = batch[9]
                if (input_mode == "qc"):
                    attention_mask_qve = batch[10]

                # 准备QVE模型输入
                inputs = {
                    "input_ids": batch[6],        # 问题/上下文的token IDs
                    "attention_mask": attention_mask_qve,   # 注意力掩码
                    "token_type_ids": batch[11],    # 段落分隔标记（如BERT需要）
                    "marginal_info": batch[12],    # 边际信息（可回答性概率）
                }


                # 模型直接输出回归值，无需条件判断
                #qa_values = qve_models[input_mode](**inputs)[0].view(-1)

                if loss_weight == input_mode:
                    # 计算模型输出以取得形状
                    temp_values = qve_models[input_mode](**inputs)[0].view(-1)
                    # 用相同的形状创建全零张量
                    qa_values = torch.zeros_like(temp_values)
                else:
                    # 计算模型输出
                    qa_values = qve_models[input_mode](**inputs)[0].view(-1)
                #logger.info("分数: %s", qa_values)
                model_scores.append(qa_values)
            
            # 堆叠分数 [batch_size, num_models]
            raw_scores = torch.stack(model_scores, dim=-1)
            
            # 2. 动态权重计算最终分数
            qa_values = dynamic_weight_model(raw_scores)

            # 对回归结果应用sigmoid归一化到[0,1]
            qa_values_normalized = torch.sigmoid(qa_values)

            all_qa_values.append(qa_values_normalized)
            all_qa_answerabilities.append((1 - batch[12]).mean(dim=1))

    # 合并所有批次的评估结果
    all_qa_values = torch.cat(all_qa_values).cpu().detach().numpy()
    all_qa_answerabilities = torch.cat(all_qa_answerabilities)

    # # 将结果转为numpy数组（后续处理可能需要）
    # # all_qa_values = Binomial(1, all_qa_values).sample()
    # all_qa_values = all_qa_values.cpu().detach().numpy()

    # 聚合每个问题的最大价值得分
    qid2feature = {}
    for ii, feature in enumerate(features):
        if feature.qas_id not in qid2feature:
            qid2feature[feature.qas_id] = [ii]
        else:
            qid2feature[feature.qas_id].append(ii)

    # 聚合每个问题的最大价值得分
    qid2qv = {}
    for qid, fids in qid2feature.items():
        qid2qv[qid] = max(all_qa_values[fids])# 取同一问题的多个特征中的最高分

    # 计算筛选前的平均分（0-100）
    pre_filter_scores = list(qid2qv.values())
    pre_avg_score = (sum(pre_filter_scores) / len(pre_filter_scores)) * 100 if pre_filter_scores else 0.0

    # 按分数排序问题ID，并截取前N%
    sorted_qid_scores = sorted(qid2qv.items(), key=lambda x: x[1], reverse=True)
    selected_num = int(len(sorted_qid_scores) * args.selected_question_percentage)
    filtered_id_list = [qid for qid, _ in sorted_qid_scores[:selected_num]]

    # 计算筛选后的平均分（0-100）
    post_filter_scores = [score for qid, score in sorted_qid_scores[:selected_num]]
    post_avg_score = (sum(post_filter_scores) / len(post_filter_scores)) * 100 if post_filter_scores else 0.0

    # 计算分数阈值（入选的最低分，0-100）
    score_threshold = (sorted_qid_scores[selected_num-1][1] * 100) if selected_num > 0 else 0.0

    # 输出统计信息
    logger.info(f"筛选前平均分: {pre_avg_score:.2f} (范围 0-100)")
    logger.info(f"筛选后平均分: {post_avg_score:.2f} (范围 0-100)")
    logger.info(f"筛选阈值: {score_threshold:.2f} (分数≥此值的问题被选中)")


    # 读取原始训练文件进行过滤
    # filtered_id_list = [qid for qid,qv in qid2qv.items() if qv==1]
    ##write to json
    data_json = json.load(open(args.train_file, 'r'))
    new_passages_train = []

    # 逐层重建过滤后的数据结构
    for passages in data_json['data']:
        new_paras_train = []

        for para in passages['paragraphs']:
            context = para['context']
            new_qas_train = []

            for qa in para['qas']:
                if qa['id'] in filtered_id_list:# 保留被选中的问题
                    new_qas_train.append(qa)

            if len(new_qas_train) > 0:
                new_paras_train.append({'context': context, 'qas': new_qas_train})

        if len(new_paras_train) > 0:
            new_passages_train.append({'title': passages['title'], 'paragraphs': new_paras_train})

    # 构建最终过滤后的数据对象
    filtered_data_json = {'data': new_passages_train, 'version': data_json['version']}
 
    # 统计过滤前后的数据量对比
    total = 0
    context_num = 0
    for paras in data_json['data']:
        for para in paras['paragraphs']:
            context_num += 1
            qa_num = len(para['qas'])
            total += qa_num
    logger.info('Before filtering: Train QA Num: %d, Total Context: %d' % (total, context_num))

 
    total = 0
    context_num = 0
    for paras in filtered_data_json['data']:
        for para in paras['paragraphs']:
            context_num += 1
            qa_num = len(para['qas'])
            total += qa_num
    logger.info('After filtering: Train QA Num: %d, Total Context: %d' % (total, context_num))

    # 保存过滤后的数据集
    if loss_weight == "":
        filename = "filtered_score_QVE.json"
    elif loss_weight == "ca" or loss_weight == "qc" or loss_weight == "qa":
        modes_to_include = [mode for mode in ["ca", "qc", "qa"] if mode != loss_weight]
        filename = f"filtered_score_{'_'.join(modes_to_include)}.json"
    else:
        filename = f"filtered_score_{loss_weight}.json"


    logger.info("Saving results to %s", filename)
    enhanced_output_dir = os.path.join(args.output_dir)
    json.dump(filtered_data_json, open(os.path.join(enhanced_output_dir, filename), 'w'))

    return
    

def save_initial_weights(weights, output_dir):
    # 将权重转换为字符串并用下划线连接
    weights_str = "_".join([f"{w:.2f}" for w in weights])
    # 创建完整的文件路径
    file_name = f"initial_weights_{weights_str}.pt"
    file_path = os.path.join(output_dir, file_name)
    
    return file_path

def main():
    parser = argparse.ArgumentParser()

    #参数配置
    ############################################
    # 核心路径参数配置
    ############################################
    #三个用到的模型
    parser.add_argument(
        "--qa_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    #用于生成数据集的QA模型
    parser.add_argument(
        "--qve_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained question value estimator",
    )
    #qve模型（需要被自适应训练的模型）
    parser.add_argument(
        "--marginal_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    #用于向QVE提供附加输入的边际QA模型。

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    #模型输出

    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file.",
    )
    #训练用数据集
    parser.add_argument(
        "--dev_file",
        default=None,
        type=str,
        help="The input dev file (target annotations) to provide feedback for QVE training. ",
    )
    #用于评估QA模型并提供QVE奖励的目标dev文件

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    #模型配置，如果模型名和模型配置不一致时使用
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    #tokenizer配置，如果模型名和tokenizer不一致时使用
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    #cache配置，如果模型名和cache不一致时使用

    ############################################
    # 是否用fp16加速
    ############################################
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    #使用混合精度（使用16位浮点数代替32位进行训练—）加快训练
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    #混合精度优化级别

    ############################################
    # 训练参数
    ############################################
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    #输入序列的最大token长度（含问题+文本）
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    #当将一个长文档分割成块时，块之间需要多大的步幅。

    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
             "be truncated to this length.",
    )
    #问题的最大标记数。比这更长的问题将被截断到这个长度。
    
    ############################################
    # 进行训练或评估
    ############################################
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    #是否进行训练（没有的化就单单时评估）
    parser.add_argument("--do_estimation", action="store_true", help="Whether to question value estimation for the training set")
    #是否进行功能评估

    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    #将输入文本统一转为小写。应对部分只能用小写的模型

    parser.add_argument(
        "--gradient_checkpointing", action="store_true", help="Set this flag if you are using gradient checkpointing"
    )
    #如果使用渐变检查点（500，1000，1500检查点）

    ############################################
    # 强化学习参数
    ############################################
    parser.add_argument("--per_gpu_train_qve_batch_size", default=60, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_train_qa_batch_size", default=6, type=int, help="Batch size per GPU/CPU for QA training.")
    #预训练qve模型和qa模型的batch_size

    parser.add_argument("--reward_type", default="exact", type=str, help="reward type: exact/f1/loss")
    parser.add_argument("--sliding_window_size", default=5, type=int,
                        help="sliding window size for updating baseline of reinforced algo.")
    #奖励类型
    #滑动窗口算法
    ############################################
    #类型	计算方式	适用场景
    #exact	精确匹配得分	答案确定性高的任务
    #f1	F1分数	开放域问答
    #loss	模型预测的负损失值	探索性训练
    ############################################

    parser.add_argument(
        "--per_gpu_eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--qve_learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--qve_eval_data_num", default=-1, type=int,
                        help="how many target data used for calculating reinforced loss for qve. -1 means all")
    #用于评估的每GPU/CPU批量大小
    #qa模型学习率
    #qve学习率
    #qve用多少数据

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    #梯度累积步数（模拟更大batch size）  通过累积多个小batch的梯度来模拟大batch训练
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    #L2正则化系数
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    #Adam优化器稳定项
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    #梯度最大范数
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    #训练轮数
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    # 最大训练步数
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    #学习率预热步数

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    #输出日志步数
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    #模型保存步数

    parser.add_argument(
        "--selected_question_percentage", default=0.6, type=float, help="how many questions to select?"
    )
    #选取问题的百分比

    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    #禁用cuda

    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    #是否对输出和cache进行复写

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    #随机种子 （很浪漫啊）

    parser.add_argument("--threads", type=int, default=24, help="multiple threads for converting example to features")
    #高性能 多线程

    parser.add_argument("--add_marginal_info", action="store_true", help="Whether not to add marginal info to qve model")
    #是否向QVE模型注入边际信息

    # 添加命令行参数：可选的检查点目录
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="可选的检查点目录")

    # 添加命令行参数：最小权重值
    parser.add_argument("--min_weight", type=float, default=0.05, help="最小权重值")

    # 添加命令行参数：调整权重的训练周期数
    parser.add_argument("--weight_epochs", type=int, default=10, help="调整权重的训练周期数")

    # 添加命令行参数：权重更新的学习率
    parser.add_argument("--weight_lr", type=float, default=1e-3, help="权重更新的学习率")

    # 添加命令行参数：权重衰减（L2正则化）
    parser.add_argument("--dy_weight_decay", type=float, default=1e-5, help="权重衰减（L2正则化）")

    # 添加命令行参数：熵系数
    parser.add_argument("--entropy_coef", type=float, default=0.1, help="熵系数")

    parser.add_argument("--weight_file", type=str, default=None, help="动态权重")

    parser.add_argument("--weight_temp", type=float, default=0.3, 
                    help="动态权重模型的温度参数")

    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )
    #防止文档分割步长过大导致特征构建失败

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    #防止意外覆盖已有结果

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    args.device = device
    #统一日志格式并输出关键配置信息

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "device: %s, n_gpu: %s, fp16: %s",
        device,
        args.n_gpu,
        args.fp16,
    )

    # Set seed
    set_seed(args)

    ############################################
    # 训练流程
    ############################################
    if args.do_train:

        # 初始化分词器（使用QVE模型的tokenizer）
        args.reward_type = args.reward_type.lower()

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.qve_model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        # 加载三个核心模型
        # 1. 边际QA模型（用于辅助QVE训练）
        # 2. 主QA模型（需要训练优化的问答模型）
        # 3. QVE模型（问题价值评估模型）
        config = AutoConfig.from_pretrained(
            args.marginal_model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        qa_model_marginal = BertForQuestionAnswering.from_pretrained(
            args.marginal_model_name_or_path,
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        config = AutoConfig.from_pretrained(
            args.qa_model_name_or_path,
            gradient_checkpointing=args.gradient_checkpointing,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        qa_model = BertForQuestionAnswering.from_pretrained(
            args.qa_model_name_or_path,
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        #在两种模式下循环训练
        for mode in ["ca","qc", "qa"]:    
        #for mode in ["qa"]:
            logger.info(f"Starting training for mode: {mode}")

            config = AutoConfig.from_pretrained(
                args.qve_model_name_or_path,
                num_labels=1,
                gradient_checkpointing=args.gradient_checkpointing,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            config.marginal = args.add_marginal_info  # add marginal information for QVE

            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name if args.tokenizer_name else args.qve_model_name_or_path,
                do_lower_case=args.do_lower_case,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            
            # === 重置qve_model为初始状态 ===
            qve_model = BertForSequenceClassification.from_pretrained(
                args.qve_model_name_or_path,
                from_tf=bool(".ckpt" in args.qve_model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )

            if '<ANS>' not in tokenizer.additional_special_tokens:#检查分词器是否包含某个token
                special_tokens_dict = {'additional_special_tokens': ['<ANS>', '<NULL_ANS>']}#定义需要的特殊标记
                tokenizer.add_special_tokens(special_tokens_dict)#将自定义标记添加到分词器的词汇表中
                logger.info("Adding Special Tokens: %s", special_tokens_dict)
            
            qve_model.resize_token_embeddings(len(tokenizer))#定义token大小
            
            # === 将模型移动到设备 ===
            qa_model.to(args.device)
            qve_model.to(args.device)
            qa_model_marginal.to(args.device)
            
            # === 训练流程 ===
            input_mode = mode  # 设置当前训练模式
            logger.info("Training/evaluation parameters %s", args)
            
            # 加载训练集和验证集
            train_dataset = load_and_cache_3p_examples(args, tokenizer, qa_model_marginal, output_examples=False, dev=False)
            dev_dataset = load_and_cache_examples(args, tokenizer, output_examples=False, dev=True)
            
            # 核心训练
            train_model(args, train_dataset, dev_dataset, input_mode, qve_model, qa_model, tokenizer)

            # === 保存当前模式的模型 ===
            enhanced_output_dir = os.path.join(args.output_dir + input_mode)
            logger.info("Saving model checkpoint to %s", enhanced_output_dir)
            
            model_to_save = qve_model.module if hasattr(qve_model,"module") else qve_model
            model_to_save.save_pretrained(enhanced_output_dir)
            tokenizer.save_pretrained(enhanced_output_dir)
            torch.save(args, os.path.join(enhanced_output_dir, "training_args.bin"))
            
            logger.info(f"Completed training for mode: {input_mode})")

        ##########################################
        #动态权重训练
        ##########################################
        # logger.info("开始固定权重保存")
        # initial_weights_list = [
        #     [0.8, 0.1, 0.1],
        #     [0.1, 0.8, 0.1],
        #     [0.1, 0.1, 0.8],
        #     [0.33, 0.33, 0.34],
        # ]
        # for initial_weights in initial_weights_list:
        #     # 调用函数保存权重
        #     saved_weights_path = save_initial_weights(initial_weights, args.output_dir)
        #     weight = DynamicWeightModel(init_weights=initial_weights)
        #     weight.save(saved_weights_path)
        #     logger.info("初始权重 %s 已保存于: %s", initial_weights, saved_weights_path)
        # logger.info("固定权重保存完成")


        logger.info("开始动态权重训练")
        logger.info("加载QVE模型和分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.qve_model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        if '<ANS>' not in tokenizer.additional_special_tokens:#检查分词器是否包含某个token
            special_tokens_dict = {'additional_special_tokens': ['<ANS>', '<NULL_ANS>']}#定义需要的特殊标记
            tokenizer.add_special_tokens(special_tokens_dict)#将自定义标记添加到分词器的词汇表中
            logger.info("Adding Special Tokens: %s", special_tokens_dict)

        qve_models = {}
        for mode in ["ca","qc", "qa"]:
            enhanced_output_dir = os.path.join(args.output_dir  + mode)
            config = AutoConfig.from_pretrained(
                enhanced_output_dir,
                num_labels=1,
                gradient_checkpointing=False,
            )#加载模型地址和禁用梯度检查点（节约内存）
            config.marginal = args.add_marginal_info  # add marginal information for QVE

            qve_models[mode] = BertForSequenceClassification.from_pretrained(enhanced_output_dir,config=config)
            qve_models[mode].resize_token_embeddings(len(tokenizer))#定义token大小
            qve_models[mode].to(args.device)
            logger.info("加载 %s 模型完成 (%s 参数)", mode, mode)

        qa_model = BertForQuestionAnswering.from_pretrained(args.qa_model_name_or_path)
        qa_model.to(args.device)
        logger.info("QA模型加载完成")

        qa_model_marginal.to(args.device)

        logger.info("加载训练和验证数据集...")
        train_dataset = load_and_cache_3p_examples(args, tokenizer, qa_model_marginal, output_examples=False, dev=False)
        dev_dataset = load_and_cache_examples(args, tokenizer, output_examples=False, dev=True)

        # 初始化动态权重
        logger.info("初始化动态权重...")

        #创建新的权重模型
        dynamic_weight_model = DynamicWeightModel(
            num_models=len(qve_models), 
            init_temp=args.weight_temp
        ).to(args.device)

        dynamic_weight_model = DynamicWeightModel(
            init_weights=[0.36,0.32,0.32], 
            init_temp=args.weight_temp,
        ).to(args.device)

        #dynamic_weight_model = DynamicWeightModel(num_models=len(qve_models), init_weights="random").to(args.device)


        # 获取并记录初始权重
        initial_weights = dynamic_weight_model.normalized_weights.detach().cpu().numpy()
        logger.info("初始权重: %s (温度: %.3f)", initial_weights, args.weight_temp)

        # 设置优化器
        optimizer_weights = torch.optim.Adam(
            dynamic_weight_model.parameters(),
            lr=args.weight_lr,
            weight_decay=args.dy_weight_decay
        )

        logger.info("开始权重训练...")
        trained_weight_model = train_weights(
            args=args,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            qve_models=qve_models,
            qa_model=qa_model,
            optimizer_weights=optimizer_weights,
            tokenizer=tokenizer,
            dynamic_weight_model=dynamic_weight_model  # 传入初始化后的权重模型
        )

        final_path = os.path.join(args.output_dir, "final_weights.pt")
        trained_weight_model.save(final_path)
        final_weights = trained_weight_model.normalized_weights.detach().cpu().numpy()
        logger.info("训练完成！最终权重保存在: %s", final_path)
        logger.info("最终权重: %s", final_weights)
    


    # ############################################
    # #  评估
    # ############################################
    if args.do_estimation:
        # ###########################################
        # #单一价值评估流程
        # ###########################################
        # for train_mode in ["ca","qc", "qa"]:
        # # for train_mode in ["ca"]:
        # #train_mode = "ca"
        #     if args.do_train:#训练模式
        #         logger.info("Loading checkpoints saved during training for evaluation")
        #         enhanced_output_dir = os.path.join(args.output_dir  + train_mode)
        #         checkpoints = enhanced_output_dir
        #     else:#测试模式
        #         enhanced_output_dir = os.path.join(args.output_dir  + train_mode)
        #         logger.info("Loading checkpoint %s for evaluation", enhanced_output_dir)
        #         #checkpoints = args.qve_model_name_or_path
        #         checkpoints = enhanced_output_dir

        #     config = AutoConfig.from_pretrained(
        #         checkpoints,
        #         num_labels=1,
        #         gradient_checkpointing=False,
        #     )#加载模型地址和禁用梯度检查点（节约内存）

        #     #加载分词器
        #     tokenizer = AutoTokenizer.from_pretrained(
        #         args.tokenizer_name if args.tokenizer_name else enhanced_output_dir,#分词器名
        #         do_lower_case=args.do_lower_case,#是否区分大小写
        #         cache_dir=args.cache_dir if args.cache_dir else None,#分词器缓存目录
        #     )

        #     # 加载评估用模型
        #     qve_model = BertForSequenceClassification.from_pretrained(checkpoints,config=config)
        #     qa_model = BertForQuestionAnswering.from_pretrained(args.qa_model_name_or_path)

        #     # 设备部署
        #     qa_model.to(args.device)
        #     qve_model.to(args.device)

        #     # 执行评估
        #     estimation(args, tokenizer, qve_model, train_mode, qa_model)

        ###########################################
        #价值评估流程
        ###########################################
        logger.info("多维价值评估流程")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else os.path.join(args.output_dir + "ca"),
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        qve_models = {}
        for mode in ["ca","qc", "qa"]:
            if args.do_train:#训练模式
                enhanced_output_dir = os.path.join(args.output_dir + mode)
            else:#测试模式
                enhanced_output_dir = os.path.join(args.qve_model_name_or_path  + mode)

            config = AutoConfig.from_pretrained(
                enhanced_output_dir,
                num_labels=1,
                gradient_checkpointing=False,
            )#加载模型地址和禁用梯度检查点（节约内存）
            config.marginal = args.add_marginal_info  # add marginal information for QVE

            qve_models[mode] = BertForSequenceClassification.from_pretrained(enhanced_output_dir,config=config)
            qve_models[mode].resize_token_embeddings(len(tokenizer))#定义token大小
            qve_models[mode].to(args.device)
            logger.info("加载 %s 模型完成 (%s 参数)", mode, mode)
        logger.info("QVE模型加载完成")

        qa_model = BertForQuestionAnswering.from_pretrained(args.qa_model_name_or_path)
        qa_model.to(args.device)
        logger.info("QA模型加载完成")

        if args.do_train:#训练模式
            weight_file = os.path.join(args.output_dir, "final_weights.pt")
        else:#测试模式
            weight_file = args.weight_file
        logger.info("weight_file: %s", weight_file)
        dynamic_weight_model = DynamicWeightModel.load(weight_file, device=args.device)
        dynamic_weight_model.to(device)
        logger.info("权重加载完成")

        # #执行评估
        # for mode in ["ca","qc", "qa", ""]:
        #     estimation_3p(args = args, 
        #         tokenizer = tokenizer, 
        #         qve_models = qve_models,
        #         qa_model = qa_model,
        #         dynamic_weight_model = dynamic_weight_model,
        #         loss_weight = mode)
            
        estimation_3p(args = args, 
            tokenizer = tokenizer, 
            qve_models = qve_models,
            qa_model = qa_model,
            dynamic_weight_model = dynamic_weight_model,
            loss_weight = "")

        # initial_weights_list = [
        #     [0.8, 0.1, 0.1],
        #     [0.1, 0.8, 0.1],
        #     [0.1, 0.1, 0.8],
        #     [0.33, 0.33, 0.34],
        # ]
        # for weight in initial_weights_list:
        #     weight_str = "_".join([f"{float(w):.2f}" for w in weight])  # 将权重列表转为字符串，用下划线连接
        #     save_dir = args.output_dir if args.do_train else args.weight_file
            
        #     weights_path = save_initial_weights(weight, save_dir)
        #     logger.info("weight_file: %s", weights_path)
        #     dynamic_weight_model = DynamicWeightModel.load(weights_path, device=args.device)
        #     dynamic_weight_model.to(device)
        #     logger.info("权重加载完成")

        #     estimation_3p(args = args, 
        #         tokenizer = tokenizer, 
        #         qve_models = qve_models,
        #         qa_model = qa_model,
        #         dynamic_weight_model = dynamic_weight_model,
        #         loss_weight = weight_str)


        

if __name__ == "__main__":
    main()
