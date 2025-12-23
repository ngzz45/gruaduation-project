#!/bin/bash


# # run mmhal eval
# CUDA_VISIBLE_DEVICES=0 \
# python chair_eval.py \
# --model "llava-1.5" \
# --num-beams 4 \
# --seed 42 \
# --vti \
# --alpha_text 0.4 \
# --alpha_image 0.4 \
# --layers_range 7 28 \
# --dynamic \
# # --vsv \
# # --vsv-lambda 0.17 \
# # --logits-aug \
# # --logits-alpha 0.3 \

# read the result file
python chair_ans.py \
--cap_file '/home/common/nizz/code/VISTA/MYexp_results/chair_eval/llava-1.5/seed42_vti_alpha_text_0.4_dynamic_layers_range_7_28_beam4_max_new_tokens_512.jsonl' \
