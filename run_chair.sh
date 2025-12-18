#!/bin/bash


# # run mmhal eval
# CUDA_VISIBLE_DEVICES=0 \
# python chair_eval.py \
# --model "instructblip" \
# --num-beams 4 \
# --vti \
# --alpha_text 0.4 \
# --alpha_image 0.4 \
# # --vsv \
# # --vsv-lambda 0.17 \
# # --logits-aug \
# # --logits-alpha 0.3 \

# read the result file
python chair_ans.py \
--cap_file '/home/common/nizz/code/VISTA/VTIexp_results/chair_eval/llava-1.5/seed42_vti_beam4_max_new_tokens_512.jsonl' \
