#!/bin/bash

set -e


save_dir=/home/ubuntu/ZimingWang/N_step_TD3_Remote/Genet_new/results_2/abr
video_size_file_dir=/home/ubuntu/ZimingWang/N_step_TD3_Remote/Genet_new/data/abr/video_sizes
val_trace_dir=/home/ubuntu/ZimingWang/N_step_TD3_Remote/Genet_new/data/abr/4G-test
total_epoch=90000
train_name=udr3
config_file=/home/ubuntu/ZimingWang/N_step_TD3_Remote/Genet_new/config/abr/${train_name}.json
train_trace_dir=/home/ubuntu/ZimingWang/N_step_TD3_Remote/Genet_new/data/abr/4G-train
pretrained_model=/home/ubuntu/ZimingWang/N_step_TD3_Remote/Genet_new/results/abr/udr_valtrace_11bit/seed_30/model_saved/nn_model_ep_70000.ckpt

seed=30
#for seed in 10 20 30; do
    python  /home/ubuntu/ZimingWang/N_step_TD3_Remote/Genet_new/src/simulator/abr_simulator/pensieve/train.py  \
        --save-dir ${save_dir}/pensieve/lin \
        --exp-name ${train_name} \
        --seed ${seed} \
        --total-epoch ${total_epoch} \
        --video-size-file-dir ${video_size_file_dir} \
        --nagent 10 \
        udr \
        --real-trace-prob 1 \
        --train-trace-dir ${train_trace_dir} \
        --config-file ${config_file} \
        --val-trace-dir ${val_trace_dir}
#done
