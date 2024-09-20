#!/bin/bash

set -e

save_dir=/home/ubuntu/ZimingWang/N_step_TD3_Remote/Genet_new/results_2/abr
video_size_file_dir=/home/ubuntu/ZimingWang/N_step_TD3_Remote/Genet_new/data/abr/video_sizes
val_trace_dir=/home/ubuntu/ZimingWang/N_step_TD3_Remote/Genet_new/data/abr/NewFile-HighDensity-CUHK-test
config_file=/home/ubuntu/ZimingWang/N_step_TD3_Remote/Genet_new/config/abr/udr3.json
pretrained_model1=/home/ubuntu/ZimingWang/N_step_TD3_Remote/Genet_new/results_2/abr/pensieve/log/model_saved/nn_model_ep_40000.ckpt
train_trace_dir=/home/ubuntu/ZimingWang/N_step_TD3_Remote/Genet_new/data/abr/NewFile-HighDensity-CUHK-train
real_trace_prob=0.3

seed=30
#for seed in  10 20 30; do
    python /home/ubuntu/ZimingWang/N_step_TD3_Remote/Genet_new/src/simulator/abr_simulator/pensieve/genet_10bit.py \
        --log \
        --save-dir ${save_dir}/genet_10bit/log \
        --heuristic mpc \
        --seed ${seed} \
        --video-size-file-dir ${video_size_file_dir} \
        --config-file ${config_file} \
        --model-path  ${pretrained_model1} \
        --train-trace-dir ${train_trace_dir} \
        --real-trace-prob ${real_trace_prob} \
        --val-trace-dir ${val_trace_dir}
#done
