# ARL-MDFF: Adaptive Reinforcement Learning-based Multi-Dimensional Data Filtering Framework

## 📋 Overview

ARL-MDFF is a comprehensive framework for actionable code smell detection, consisting of four main components:

## 1. Dataset
The dataset can be downloaded using `./download_and_process.sh`, or you can directly use the data placed in the `data/` folder.

## 2. QG (Question Generation)
The QG model is used for data augmentation. It can be invoked by running `./run_qg.sh`.

## 3. QA (Question Answering model)
This module contains code related to the QA model, including performance evaluation and training. Use `run_qa_baseline.sh` to train downstream models and evaluate the final model's capabilities.

## 4. ARF (Adaptive Reinforcement Filtering)
Train and use the filtering model. Experiments (RQ1-RQ4) can be conducted through `ARF/run_arf.py`.
