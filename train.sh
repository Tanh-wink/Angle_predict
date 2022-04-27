#!/bin/bash

unset CUDA_VISIBLE_DEVICES

python -u -m paddle.distributed.launch --gpus "0,1,2" train.py \
       --train_set /home/th/paddle/question_matching/data/train_merge.txt \
       --dev_set /home/th/paddle/question_matching/data/dev.txt \
       --device gpu \
       --eval_step 300 \
       --save_dir ./checkpoints \
       --train_batch_size 20 \
       --learning_rate 2E-5 \
       --rdrop_coef 0.1
