#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# MODEL="Qwen/Qwen-7B-Chat-Int4" # Set the path if you do not want to load from huggingface directly
# MODEL="/home/data/shared/checkpoints/modelscope/Qwen-14B-Chat-Int4"
# MODEL="/home/data/yuexiang.xyx/test_ppl/Tongyi-Finance-14B-Chat-4bit"
MODEL="/home/data/shared/checkpoints/boshi_competition/models/Tongyi-Finance-14B-Chat-Int4"
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA=""
OUTPUT_DIR=""

export CUDA_VISIBLE_DEVICES=0

# Remember to use --fp16 instead of --bf16 due to autogptq
python finetune.py \
  --model_name_or_path $MODEL \
  --data_path $DATA \
  --fp16 True \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 10 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 512 \
  --lazy_preprocess True \
  --gradient_checkpointing \
  --use_lora \
  --q_lora \
  --deepspeed finetune/ds_config_zero2.json
