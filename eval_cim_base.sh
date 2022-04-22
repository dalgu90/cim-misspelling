#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,1

exp_base="cim_base"
dataset=mimic_synthetic

dict_file=data/lexicon/lexicon.json
mimic_csv_dir=data/mimic3/split
output_dir=results/$exp_base
init_step=475000

data_dir=data/$dataset
test_dir=data/cspell
test_file=test.tsv
#test_file=train.tsv
#test_dir=data/mimic_clinspell
#test_file=test.tsv

edit_distance_weight=5.0  # (2.0 5.0 10.0 20.0)
edit_distance_extra_len=1  # (100 2 1 0)
beam_sort_linear_ed="beam_sort_ed"
beam_final_score_normalize_ed="beam_final_score_normalize_ed"
length_penalty=1.0
num_beams=30

# Evaluate on the MIMIC-Clinspell or CSpell dataset
python run.py \
    --is_eval \
    --data_dir=$test_dir \
    --test_file=$test_file \
    --dict_file=$dict_file \
    --bert_dir=bert/ncbi_bert_base \
    --output_dir=$output_dir \
    --decoder_layers=12 \
    --batch_size=8 \
    --num_gpus=2 \
    --num_beams=$num_beams \
    --edit_distance_weight=$edit_distance_weight \
    --edit_distance_extra_len=$edit_distance_extra_len \
    --length_penalty=$length_penalty \
    --$beam_sort_linear_ed \
    --$beam_final_score_normalize_ed \
    --dict_matching \
    --init_ckpt=$output_dir/ckpt-${init_step}.pkl \
    --init_step=${init_step} \

# Evaulate on the val set of synthetic typo dataset
#python run.py \
    #--is_eval \
    #--data_dir=$data_dir \
    #--test_file=val.tsv \
    #--dict_file=$dict_file \
    #--bert_dir=bert/ncbi_bert_base \
    #--output_dir=$output_dir \
    #--decoder_layers=12 \
    #--batch_size=8 \
    #--num_gpus=2 \
    #--num_beams=$num_beams \
    #--edit_distance_weight=$edit_distance_weight \
    #--edit_distance_extra_len=$edit_distance_extra_len \
    #--length_penalty=$length_penalty \
    #--$beam_sort_linear_ed \
    #--$beam_final_score_normalize_ed \
    #--dict_matching \
    #--init_ckpt=$output_dir/ckpt-${init_step}.pkl \
    #--init_step=${init_step} \
