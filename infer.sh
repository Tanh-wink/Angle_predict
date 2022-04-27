#!/bin/bash

unset CUDA_VISIBLE_DEVICES

test_set="/home/th/paddle/question_matching/data/test.tsv"

python -u \
    predict.py \
    --device gpu \
    --params_path "./checkpoints/model_29400/model_state.pdparams" \
    --batch_size 256 \
    --input_file "${test_set}" \
    --result_file "./data/predict_result.csv"