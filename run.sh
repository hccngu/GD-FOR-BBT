#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

task_name_lst=(sst2)
learning_rate_lst=(1e-3)
loss_func=MSE


for task_name in "${task_name_lst[@]}"; do
    for learning_rate in "${learning_rate_lst[@]}"; do
        python -u new_bbt.py \
                  --task_name $task_name \
                  --learning_rate $learning_rate \
                  --loss_func $loss_func \
                  --n_prompt_tokens 50 \
                  --intrinsic_dim 500 \
                  --k_shot 16 \
                  --device "cuda:0" \
                  --seed 42 \
                  --loss_type "ce" \
                  --cat_or_add "add" \
                  --budget 8000 \
                  --print_every 50 \
                  --eval_every 100
    done
done
# python -u main_copy.py --tripletloss_mode sig+dp+dn --use_proto_as_neg --model MTNet --dataset fewnerd --dataset_mode IO --mode intra --trainN 10 --N 10 --K 1 --Q 1 --neg_num 1 --trainable_margin_init 8.0 --trainable_alpha_init 1.0 --dropout 0.0 --bert_lr 2e-5 --meta_lr 5e-4 --task_lr 1e-1 --train_support_iter 3 --ln_bias 10 --bert_wd 1e-5 --wobert_wd 1e-5 --train_iter 4000 --test_iter 500
endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 
