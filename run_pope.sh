#!/bin/bash

# run pope eval
CUDA_VISIBLE_DEVICES=0 \
python pope_eval.py \
--model "llava-1.5" \
--data-path "/data/nizz/coco2014/val2014" \
--pope-type 'popular' \
--seed 600 \
--num-beams 4 \
--subset-size 600 \
--vti \
--alpha_text 0.4 \
--alpha_image 0.4 \
--dynamic \
# --vsv \
# --vsv-lambda 0.01 \
# --logits-aug \
# --logits-alpha 0.3 \





# read the result file
python pope_ans.py \
--ans_file '/home/common/nizz/code/VISTA/MYexp_results/pope_eval/llava-1.5/seed600_popular_vti_alpha_text_0.4_dynamic_beam4_max_new_tokens_32.jsonl' \