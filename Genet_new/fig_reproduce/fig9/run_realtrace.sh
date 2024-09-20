#! /bin/bash
# This file runs rl_test.py on a specified model.
# immediately exit the bash if an error encountered
set -e

SIMULATOR_DIR="../sim"


# new models

NN_MODELS_UDR_3="/home/ubuntu/Whr/EAS/Genet_new/results/abr/udr_valtrace_1/seed_30/model_saved/nn_model_ep_51300.ckpt"
#NN_MODELS_UDR_3="../data/all_models/udr_3/nn_model_ep_58000.ckpt"

#NN_MODELS_ADR="/home/ubuntu/Whr/Genet_new/results5/abr/genet_mpc/seed_30/bo_5/model_saved/nn_model_ep_8800.ckpt"
#NN_MODELS_ADR="/home/ubuntu/Whr/Genet_new/results5/abr/genet_mpc2/seed_30/bo_5/model_saved/nn_model_ep_9100.ckpt"
#NN_MODELS_ADR="/home/ubuntu/Whr/EAS/Genet_new/results5/abr/genet_mpc5/seed_10/bo_0/model_saved/nn_model_ep_100.ckpt"
NN_MODELS_ADR="/home/ubuntu/Whr/EAS/Genet_new/results/abr/genet_mpc_valtrace/seed_30/bo_10/model_saved/nn_model_ep_8500.ckpt"


#TRACE_PATH="../data/synthetic_test/"
TRACE_PATH="/home/ubuntu/Whr/EAS/Genet_new/test_file"
SUMMARY_DIR="../sigcomm_artifact/synthetic/"

LOG_STR="non"

for seed in 1 2 3 4 5 6 7 8 9 10 11; do
  python ${SIMULATOR_DIR}/rl_test_clean.py \
       --test_trace_dir ${TRACE_PATH}/${seed}/ \
       --summary_dir ${SUMMARY_DIR}\
       --model_path ${NN_MODELS_ADR} \
       --log_str ${LOG_STR} \
       --random_seed=1


  python ${SIMULATOR_DIR}/rl_test_clean_udr_3.py \
       --test_trace_dir ${TRACE_PATH}/${seed}/ \
       --summary_dir ${SUMMARY_DIR}\
       --model_path ${NN_MODELS_UDR_3} \
       --log_str ${LOG_STR} \
       --random_seed=1

  python plot_results_realtrace.py \
       --seed ${seed}
done

