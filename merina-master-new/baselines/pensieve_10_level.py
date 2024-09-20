import os
import sys
#os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import tensorflow as tf
import a3c
import argparse

sys.path.append('../envs/')
import fixed_env_log_10bit as env
import load_trace
from functools import reduce
from operator import mul


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 16  # take how many frames in the past
A_DIM = 10
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [145, 365 ,730, 1100, 2000, 3000, 4500, 6000, 7800, 10000]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 150
M_IN_K = 1000.0
REBUF_PENALTY_lin = 6 #dB
REBUF_PENALTY_log = 4.23
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
#NN_MODEL = '../models/baselines/pensieve/pretrain_linear_reward.ckpt'
NN_MODEL_LOG = '/home/ubuntu/Whr/EAS/Genet_new/results_2/abr/pensieve/log/model_saved/nn_model_ep_79600.ckpt'
NN_MODEL_LIN = '/home/ubuntu/Whr/EAS/Genet_new/results_2/abr/pensieve/lin/model_saved/nn_model_ep_82700.ckpt'
#NN_MODEL= None
LOG_FILE_OBE = '../Results/test/lin/oboe/log_test_a3c'
LOG_FILE_3GP = '../Results/test/lin/3gp/log_test_a3c'
LOG_FILE_FCC = '../Results/test/lin/fcc/log_test_a3c'
LOG_FILE_FH = '../Results/test/lin/fh/log_test_a3c'
LOG_FILE_PUF = '../Results/test/lin/puffer/log_test_a3c'
LOG_FILE_PUF2 = '../Results/test/lin/puffer2/log_test_a3c'
TEST_LOG_FILE_Newfile = '../Results_10bit/test/lin/Newfile_CUHK/'

LOG_FILE_OBE_LOG = '../Results/test/log/oboe/log_test_a3c'
LOG_FILE_3GP_LOG = '../Results/test/log/3gp/log_test_a3c'
LOG_FILE_FCC_LOG = '../Results/test/log/fcc/log_test_a3c'
LOG_FILE_FH_LOG = '../Results/test/log/fh/log_test_a3c'
LOG_FILE_PUF_LOG = '../Results/test/log/puffer/log_test_a3c'
LOG_FILE_PUF2_LOG = '../Results/test/log/puffer2/log_test_a3c'
TEST_LOG_FILE_Newfile_LOG = '../Results_10bit/test/log/Newfile_CUHK/'

TEST_TRACES_FCC = '../envs/traces/fcc_ori/test_traces/'
TEST_TRACES_OBE = '../envs/traces/traces_oboe/'
TEST_TRACES_3GP = '../envs/traces/traces_3gp/'
TEST_TRACES_FH = '../envs/traces/fcc_and_hsdpa/cooked_test_traces/'
TEST_TRACES_PUF = '../envs/traces/puffer_211017/test_traces/'
TEST_TRACES_PUF2 = '../envs/traces/puffer_220218/test_traces/'
TEST_TRACES_Newfile = '../envs/traces/test_file/'


parser = argparse.ArgumentParser(description='Pensieve')
parser.add_argument('--log', action='store_true', help='Use logarithmic form QoE metric')
parser.add_argument('--tf', action='store_true', help='Use FCC traces')
parser.add_argument('--tfh', action='store_true', help='Use FCC&3GP traces')
parser.add_argument('--to', action='store_true', help='Use Oboe traces')
parser.add_argument('--t3g', action='store_true', help='Use 3GP traces')
parser.add_argument('--tp', action='store_true', help='Use Puffer traces')
parser.add_argument('--tp2', action='store_true', help='Use Puffer2 traces')
parser.add_argument('--tnf', action='store_true', help='Use Newfile traces')

 
def get_num_params():
    num_params = 0
    for variable in tf.compat.v1.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

def main():
    args = parser.parse_args()
    video = 'Mao'
    # video = 'Avengers'
    video_size_file = '../envs/video_size/' + video + '/video_size_'

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    # -----------------------------initialize the environment----------------------------------------
    if args.tf:
        test_traces = TEST_TRACES_FCC
        log_file_init = LOG_FILE_FCC_LOG if args.log else LOG_FILE_FCC
    elif args.tfh:
        test_traces = TEST_TRACES_FH
        log_file_init = LOG_FILE_FH_LOG if args.log else LOG_FILE_FH
    elif args.to:
        test_traces = TEST_TRACES_OBE
        log_file_init = LOG_FILE_OBE_LOG if args.log else LOG_FILE_OBE
    elif args.t3g:
        test_traces = TEST_TRACES_3GP
        log_file_init = LOG_FILE_3GP_LOG if args.log else LOG_FILE_3GP
    elif args.tp:
        test_traces = TEST_TRACES_PUF
        log_file_init = LOG_FILE_PUF_LOG if args.log else LOG_FILE_PUF
    elif args.tp2:
        test_traces = TEST_TRACES_PUF2
        log_file_init = LOG_FILE_PUF2_LOG if args.log else LOG_FILE_PUF2
    elif args.tnf:
        log_file_init = TEST_LOG_FILE_Newfile_LOG if args.log else TEST_LOG_FILE_Newfile
        test_traces = TEST_TRACES_Newfile
    else:
        # print("Please choose the throughput data traces!!!")
        test_traces = TEST_TRACES_FCC
        log_file_init = LOG_FILE_FCC_LOG #if args.log else LOG_FILE_FCC




    i = 10
    for ii in range(i):
        # determine the QoE metric \
        tf.keras.backend.clear_session()
        rebuff_p = REBUF_PENALTY_log if args.log else REBUF_PENALTY_lin

        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(test_traces + str(ii+1) + "/")
        test_env = env.Environment(all_cooked_time=all_cooked_time,
                                   all_cooked_bw=all_cooked_bw, all_file_names=all_file_names,
                                   video_size_file=video_size_file)

        test_env.set_env_info(0, 0, 0, int(CHUNK_TIL_VIDEO_END_CAP), \
                                VIDEO_BIT_RATE, 1, rebuff_p, SMOOTH_PENALTY, 0)

        if not os.path.exists(log_file_init+str(ii+1)+"/"):
            os.makedirs(log_file_init+str(ii+1)+"/")


        log_path = log_file_init + str(ii+1) + '/'+ 'log_test_a3c' + '_' + all_file_names[test_env.trace_idx]
        log_file = open(log_path, 'wb')

        _, _, _, total_chunk_num, \
                bitrate_versions, rebuffer_penalty, smooth_penalty = test_env.get_env_info()

        # if not os.path.exists(SUMMARY_DIR):
        #     os.makedirs(SUMMARY_DIR)

        with tf.compat.v1.Session() as sess:

            actor = a3c.ActorNetwork(sess,
                                     state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                     learning_rate=ACTOR_LR_RATE)

            critic = a3c.CriticNetwork(sess,
                                       state_dim=[S_INFO, S_LEN],
                                       learning_rate=CRITIC_LR_RATE)

            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()  # save neural net parameters

            if args.log:
                nn_model = NN_MODEL_LOG
            else:
                nn_model = NN_MODEL_LIN

            # restore neural net parameters
            if nn_model is not None:  # nn_model is the path to file
                saver.restore(sess, nn_model)
                print("Model restored.")

            num_params = 0
            for variable in tf.compat.v1.trainable_variables():
                shape = variable.get_shape()
                num_params += reduce(mul, [dim.value for dim in shape], 1)
            print ("num", num_params)

            time_stamp = 0

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch = [np.zeros((S_INFO, S_LEN))]
            a_batch = [action_vec]
            r_batch = []
            entropy_record = []

            video_count = 0

            while True:  # serve video forever
                # the action is from the last decision
                # this is to make the framework similar to the real
                delay, sleep_time, buffer_size, rebuf, \
                    video_chunk_size, next_video_chunk_sizes, \
                        end_of_video, video_chunk_remain, \
                            _ = test_env.get_video_chunk(bit_rate)

                time_stamp += delay  # in ms
                time_stamp += sleep_time  # in ms

                # reward is video quality - rebuffer penalty
                if args.log:
                    log_bit_rate = np.log(bitrate_versions[bit_rate] / \
                                            float(bitrate_versions[0]))
                    log_last_bit_rate = np.log(bitrate_versions[last_bit_rate] / \
                                                float(bitrate_versions[0]))
                    reward = log_bit_rate \
                            - rebuffer_penalty * rebuf \
                            - smooth_penalty * np.abs(log_bit_rate - log_last_bit_rate)
                else:
                    reward = bitrate_versions[bit_rate] / M_IN_K \
                            - rebuffer_penalty * rebuf \
                            - smooth_penalty * np.abs(bitrate_versions[bit_rate] -
                                                    bitrate_versions[last_bit_rate]) / M_IN_K
                r_batch.append(reward)

                last_bit_rate = bit_rate

                # log time_stamp, bit_rate, buffer_size, reward
                log_file.write((str(time_stamp / M_IN_K) + '\t' +
                               str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                               str(buffer_size) + '\t' +
                               str(rebuf) + '\t' +
                               str(video_chunk_size) + '\t' +
                               str(delay) + '\t' +
                               str(reward) + '\n').encode())
                log_file.flush()

                # retrieve previous state
                if len(s_batch) == 0:
                    state = [np.zeros((S_INFO, S_LEN))]
                else:
                    state = np.array(s_batch[-1], copy=True)

                # dequeue history record
                state = np.roll(state, -1, axis=1)

                # this should be S_INFO number of terms
                state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
                state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
                state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
                state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
                state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

                action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
                action_cumsum = np.cumsum(action_prob)
                bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / \
                                float(RAND_RANGE)).argmax()
                # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                # because there is an intrinsic discrepancy in passing single state and batch states

                s_batch.append(state)

                entropy_record.append(a3c.compute_entropy(action_prob[0]))

                if end_of_video:
                    log_file.write('\n'.encode())
                    log_file.close()

                    last_bit_rate = DEFAULT_QUALITY
                    bit_rate = DEFAULT_QUALITY  # use the default action here

                    del s_batch[:]
                    del a_batch[:]
                    del r_batch[:]

                    action_vec = np.zeros(A_DIM)
                    action_vec[bit_rate] = 1

                    s_batch.append(np.zeros((S_INFO, S_LEN)))
                    a_batch.append(action_vec)
                    entropy_record = []

                    print ("video count", video_count)
                    video_count += 1

                    if video_count >= len(all_file_names):
                        break

                    log_path = log_file_init + str(ii+1) + '/'+ 'log_test_a3c' + '_' + all_file_names[test_env.trace_idx]
                    log_file = open(log_path, 'wb')


if __name__ == '__main__':
    main()
