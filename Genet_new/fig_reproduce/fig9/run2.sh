#! /bin/bash
# This file runs rl_test.py on a specified model.
# immediately exit the bash if an error encountered
set -e

SIMULATOR_DIR="../sim"


# new models

NN_MODELS_UDR_3="/home/ubuntu/Whr/EAS/Genet_new/results/abr/udr3_3_2/seed_30/model_saved/nn_model_ep_55200.ckpt"
#NN_MODELS_UDR_3="../data/all_models/udr_3/nn_model_ep_58000.ckpt"
NN_MODELS_UDR_2="/home/ubuntu/Whr/EAS/Genet_new/results/abr/udr2_2/seed_30/model_saved/nn_model_ep_51200.ckpt"
#NN_MODELS_UDR_2="../data/all_models/udr_2/nn_model_ep_52400.ckpt"
NN_MODELS_UDR_1="/home/ubuntu/Whr/EAS/Genet_new/results/abr/udr1_2/seed_30/model_saved/nn_model_ep_49500.ckpt"
#NN_MODELS_ADR="/home/ubuntu/Whr/Genet_new/results5/abr/genet_mpc3/seed_30/bo_8/model_saved/nn_model_ep_3300.ckpt"
#NN_MODELS_ADR="/home/ubuntu/Whr/Genet_new/results5/abr/genet_mpc/seed_30/bo_5/model_saved/nn_model_ep_8800.ckpt"
#NN_MODELS_ADR="/home/ubuntu/Whr/Genet_new/results5/abr/genet_mpc2/seed_30/bo_5/model_saved/nn_model_ep_9100.ckpt"
#NN_MODELS_ADR="/home/ubuntu/Whr/EAS/Genet_new/results5/abr/genet_mpc5/seed_10/bo_0/model_saved/nn_model_ep_100.ckpt"
NN_MODELS_ADR="/home/ubuntu/Whr/EAS/Genet_new/results/abr/genet_mpc_new_3/seed_30/bo_13/model_saved/nn_model_ep_6000.ckpt"


TRACE_PATH="../data/synthetic_test/"
#TRACE_PATH="/home/ubuntu/Whr/EAS/Genet_new/test_file/6"
#TRACE_PATH="/home/ubuntu/Whr/EAS/Genet_new/data/abr/val_FCC"
#TRACE_PATH="/home/ubuntu/Whr/EAS/Genet_new/data/abr/NewFile-HighDensity-CUHK-test"
SUMMARY_DIR="../sigcomm_artifact/synthetic/"

LOG_STR="non"


python ${SIMULATOR_DIR}/rl_test_clean.py \
       --test_trace_dir ${TRACE_PATH} \
       --summary_dir ${SUMMARY_DIR} \
       --model_path ${NN_MODELS_ADR} \
       --log_str ${LOG_STR} \
       --random_seed=1


python ${SIMULATOR_DIR}/rl_test_clean_udr_1.py \
       --test_trace_dir ${TRACE_PATH} \
       --summary_dir ${SUMMARY_DIR} \
       --model_path ${NN_MODELS_UDR_1} \
       --log_str ${LOG_STR} \
       --random_seed=1

python ${SIMULATOR_DIR}/rl_test_clean_udr_2.py \
       --test_trace_dir ${TRACE_PATH} \
       --summary_dir ${SUMMARY_DIR} \
       --model_path ${NN_MODELS_UDR_2} \
       --log_str ${LOG_STR} \
       --random_seed=1

python ${SIMULATOR_DIR}/rl_test_clean_udr_3.py \
       --test_trace_dir ${TRACE_PATH} \
       --summary_dir ${SUMMARY_DIR} \
       --model_path ${NN_MODELS_UDR_3} \
       --log_str ${LOG_STR} \
       --random_seed=1

python plot_results.py \
       --seed=0


