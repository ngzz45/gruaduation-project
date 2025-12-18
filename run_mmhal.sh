#!/bin/bash

# # run mmhal eval
# CUDA_VISIBLE_DEVICES=0 \
# python mmhal_eval.py \
# --model "llava-1.5" \
# --vsv \
# --vsv-lambda 0.1 \
# --logits-aug \
# --logits-alpha 0.3 \

# read the result file
python mmhal_ans.py \
--response '/home/common/nizz/code/VISTA/exp_results/mmhal_eval/llava-1.5/seed1994_vsv_lambda_0.1_logaug_loglayer_25,30_logalpha_0.3_greedy_max_new_tokens_512.jsonl' \