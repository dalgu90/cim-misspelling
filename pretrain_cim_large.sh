#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,1,2,3

exp_base="cim_large"
dataset=mimic_synthetic

dict_file=data/lexicon/lexicon.json
mimic_csv_dir=data/mimic3/split
data_dir=data/$dataset
output_dir=results/$exp_base

# Decoding for eval
length_penalty=1.0
beam_sort_linear_ed="beam_sort_ed"
beam_final_score_normalize_ed="beam_final_score_normalize_ed"
edit_distance_weight=5.0

python run.py \
    --mimic_csv_dir=$mimic_csv_dir \
    --data_dir=$data_dir \
    --bert_dir=bert/ncbi_bert_large \
    --dict_file=$dict_file \
    --output_dir=$output_dir \
    --is_train \
    --dropout=0.1 \
    --decoder_layers=24 \
    --train_bert \
    --bert_finetune_factor=0.1 \
    --batch_size=256 \
    --num_gpus=8 \
    --num_beams=3 \
    --edit_distance_weight=$edit_distance_weight \
    --length_penalty=$length_penalty \
    --$beam_sort_linear_ed \
    --$beam_final_score_normalize_ed \
    --dict_matching \
    --training_step=500000 \
    --display_iter=100 \
    --eval_iter=25000 \
    --lr 0.0001 \
    #--init_ckpt=$output_dir/ckpt.pkl \
    #--init_step=100000
