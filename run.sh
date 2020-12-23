#!/usr/bin/env bash

export MID_DATA_DIR="./data/mid_data"
export RAW_DATA_DIR="./data/raw_data"
export OUTPUT_DIR="./out"

export GPU_IDS="0"
export BERT_TYPE="roberta_wwm"  # roberta_wwm / roberta_wwm_large / uer_large
export BERT_DIR="../bert/torch_$BERT_TYPE"

export MODE="train"
export TASK_TYPE="crf"

python main.py \
--gpu_ids=$GPU_IDS \
--output_dir=$OUTPUT_DIR \
--mid_data_dir=$MID_DATA_DIR \
--mode=$MODE \
--task_type=$TASK_TYPE \
--raw_data_dir=$RAW_DATA_DIR \
--bert_dir=$BERT_DIR \
--bert_type=$BERT_TYPE \
--train_epochs=10 \
--swa_start=5 \
--attack_train="" \
--train_batch_size=24 \
--dropout_prob=0.1 \
--max_seq_len=512 \
--lr=2e-5 \
--other_lr=2e-3 \
--seed=123 \
--weight_decay=0.01 \
--loss_type='ls_ce' \
--eval_model \
#--use_fp16