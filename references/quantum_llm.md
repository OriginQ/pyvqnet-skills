# 量子大模型微调 API Reference

> 来源: VQNET2.0-tutorial/source/rst/llm.rst
> **重要**: 需要 pyvqnet >= 2.15.0

---

## 概述

量子大模型微调通过与 Llama Factory、peft 结合，实现基于量子线路进行大模型微调任务。

核心依赖：
- `quantum-llm` 库
- `pyvqnet >= 2.15.0`

---

## 安装步骤

### 1. 安装 quantum-llm

```bash
git clone https://gitee.com/craftsman_lei/quantum-llm.git

# 安装依赖
pip install -r requirements.txt

# 安装 peft_vqc
cd peft_vqc && pip install -e .
```

### 2. 安装 pyvqnet

```bash
pip install pyvqnet  # pyvqnet>=2.15.0
```

---

## 微调模块类型

| 类型 | 描述 |
|------|------|
| `vqc` | 基于 VQNet 实现的 VQC 微调模块 |
| `quanTA` | 量子张量分解模块 |
| `tq` | 基于 torch quantum 实现的 VQC 模块 |

---

## 基准模型下载

下载 Qwen2.5-0.5B：

```bash
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B
```

---

## 训练脚本 train.sh

脚本位于 `/quantum-llm/examples/qlora_single_gpu/` 目录。

```bash
#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python ../../src/train_bash.py \
    --stage sft \
    --model_name_or_path /下载路径/Qwen2.5-0.5B/ \
    --dataset alpaca_gpt4_en \
    --tokenized_path ../../data/tokenized/alpaca_gpt4_en/ \
    --dataset_dir ../../data \
    --template qwen \
    --finetuning_type vqc \
    --lora_target q_proj,v_proj \
    --output_dir ../../saves/Qwen2.5-0.5B/vqc/alpaca_gpt4_en \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 1000 \
    --val_size 0.1 \
    --plot_loss \
    --fp16 \
    --do-train

# 执行训练
sh train.sh
```

**注意**: 将 `finetuning_type` 改为 `quanTA` 或 `tq` 可切换不同微调模块。

---

## 评估脚本 eval.sh

```bash
#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python ../../src/evaluate.py \
    --model_name_or_path /下载路径/Qwen2.5-0.5B/ \
    --template qwen \
    --finetuning_type vqc \
    --task cmmlu \
    --task_dir ../../evaluation/ \
    --adapter_name_or_path ../../saves/Qwen2.5-0.5B/vqc/alpaca_gpt4_en

# 执行评估
sh eval.sh
```

支持的任务: `cmmlu`, `ceval`, `mmlu`

---

## 问答脚本 cli.sh

```bash
#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python ../../src/cli_demo.py \
    --model_name_or_path /下载路径/Qwen2.5-0.5B/ \
    --template qwen \
    --finetuning_type vqc \
    --adapter_name_or_path ../../saves/Qwen2.5-0.5B/vqc/alpaca_gpt4_en \
    --max_new_tokens 1024

# 执行问答
sh cli.sh
```

---

## 参数说明

| 参数 | 说明 |
|------|------|
| `stage` | 训练模式: `pt`(预训练), `sft`(微调) |
| `model_name_or_path` | 基准模型路径 |
| `dataset` | 数据集: `identity`, `alpaca_gpt4_zh` 等 |
| `tokenized_path` | 数据集 tokenized 路径 |
| `dataset_dir` | 数据集路径 |
| `template` | 模型模板: `qwen`, `llama3` 等 |
| `finetuning_type` | 微调方法: `lora`, `vqc`, `quanTA`, `tq` |
| `lora_target` | 作用模块: `q_proj,v_proj` |
| `output_dir` | 微调模块保存路径 |
| `overwrite_cache` | 是否覆盖缓存 |
| `overwrite_output_dir` | 是否覆盖输出目录 |
| `cutoff_len` | 数据截断长度 |
| `preprocessing_num_workers` | 预处理工作进程数 |
| `per_device_train_batch_size` | 每个 GPU 批处理大小 |
| `per_device_eval_batch_size` | 评估批次大小 |
| `gradient_accumulation_steps` | 梯度累计步数 |
| `lr_scheduler_type` | 学习率调度器 |
| `logging_steps` | 打印间隔 |
| `warmup_steps` | 预热步数 |
| `save_steps` | 模型保存间隔 |
| `eval_steps` | 评估间隔 |
| `evaluation_strategy` | 评估策略 |
| `load_best_model_at_end` | 训练结束加载最佳模型 |
| `learning_rate` | 学习率 |
| `num_train_epochs` | 训练轮数 |
| `max_samples` | 最大样本数 |
| `val_size` | 验证集大小 |
| `plot_loss` | 是否保存训练损失曲线 |
| `fp16` | 使用 fp16 混合精度（vqc 模块使用 float32） |
| `do-train` | 是否为训练任务 |
| `adapter_name_or_path` | 训练后生成文件路径 |
| `task` | 评估任务: `ceval`, `cmmlu`, `mmlu` |
| `task_dir` | 任务路径 |
| `q_d` | quanTA 模块张量分解数量（默认 4） |
| `per_dim_features` | quanTA 模块张量分解特征数（默认 [16,8,4,2]） |

---

## 实验结果

基于 Qwen2.5-0.5B 在 alpaca_gpt4_en 数据集的训练结果表明，基于 VQNet 的 `vqc` 模块取得了最好的损失收敛效果。

---

## 完整流程

```bash
# 1. 安装依赖
git clone https://gitee.com/craftsman_lei/quantum-llm.git
pip install -r requirements.txt
cd peft_vqc && pip install -e .
pip install pyvqnet

# 2. 下载基准模型
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B

# 3. 训练
cd quantum-llm/examples/qlora_single_gpu
sh train.sh

# 4. 评估
sh eval.sh

# 5. 问答
sh cli.sh
```

---

**Version**: VQNet 2.0
**Source**: VQNET2.0-tutorial/source/rst/llm.rst