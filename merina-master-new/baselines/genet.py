import argparse
import os
import a3c
# import fixed_env as env
import sys

sys.path.append('../envs/')
import fixed_env_log as env
import load_trace
import numpy as np
import tensorflow as tf
import subprocess
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR )


# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 16  # take how many frames in the past
A_DIM = 3
BITRATE_DIM=10
# S_LEN = 11  # take how many frames in the past
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
CHUNK_TIL_VIDEO_END_CAP = 150
REBUF_PENALTY = 6
REBUF_PENALTY_lin = 6 #dB
REBUF_PENALTY_log = 4.23
VIDEO_BIT_RATE = [145, 365 ,730, 1100, 2000, 3000, 4500, 6000, 7800, 10000]  # Kbps
BUFFER_NORM_FACTOR = 10.0
#CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
#REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
# RANDOM_SEED = 42
RAND_RANGE = 1000
RANDOM_SEED = 42

# LOG_FILE = './test_results/log_sim_rl'
# TEST_TRACES = './cooked_test_traces/'
# TEST_TRACES = './test_sim_traces/'
# TEST_TRACES = '../data/val/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward



NN_MODEL = '/home/ubuntu/Whr/EAS/Genet_new/results5/abr/genet_mpc_11bit/seed_30/bo_6/model_saved/nn_model_ep_4200.ckpt'
#NN_MODEL= None
LOG_FILE_OBE = '../Results/test/lin/oboe/log_test_genet'
LOG_FILE_3GP = '../Results/test/lin/3gp/log_test_genet'
LOG_FILE_FCC = '../Results/test/lin/fcc/log_test_genet'
LOG_FILE_FH = '../Results/test/lin/fh/log_test_genet'
LOG_FILE_PUF = '../Results/test/lin/puffer/log_test_genet'
LOG_FILE_PUF2 = '../Results/test/lin/puffer2/log_test_genet'
TEST_LOG_FILE_Newfile = '../Results_Newfile_next/test/lin/Newfile_CUHK/log_test_genet'

LOG_FILE_OBE_LOG = '../Results/test/log/oboe/log_test_genet'
LOG_FILE_3GP_LOG = '../Results/test/log/3gp/log_test_genet'
LOG_FILE_FCC_LOG = '../Results/test/log/fcc/log_test_genet'
LOG_FILE_FH_LOG = '../Results/test/log/fh/log_test_genet'
LOG_FILE_PUF_LOG = '../Results/test/log/puffer/log_test_genet'
LOG_FILE_PUF2_LOG = '../Results/test/log/puffer2/log_test_genet'
TEST_LOG_FILE_Newfile_LOG = '../Results_Newfile_next/test/log/Newfile_CUHK/log_test_genet'

TEST_TRACES_FCC = '../envs/traces/fcc_ori/test_traces/'
TEST_TRACES_OBE = '../envs/traces/traces_oboe/'
TEST_TRACES_3GP = '../envs/traces/traces_3gp/'
TEST_TRACES_FH = '../envs/traces/fcc_and_hsdpa/cooked_test_traces/'
TEST_TRACES_PUF = '../envs/traces/puffer_211017/test_traces/'
TEST_TRACES_PUF2 = '../envs/traces/puffer_220218/test_traces/'
TEST_TRACES_Newfile = '../envs/traces/merina_test_trace/'




# Strategy:

# Input for RL Testing should be:
#
# 1. a configuration from which test traces are generated
#   - load the configuration from json and create a TraceConfig to generate traces (later)
#   - create the traces from a configuration (refer to example) (priority)
#
# 2. a model checkpoint file to load and test against the traces (DONE)
# 3. Move TraceConfig outside of this file so it can be used elsewhere too. (later)

def parse_args():
    parser = argparse.ArgumentParser(description="Pensieve" )
    parser.add_argument('--log', action='store_true', help='Use logarithmic form QoE metric')
    parser.add_argument('--tf', action='store_true', help='Use FCC traces')
    parser.add_argument('--tfh', action='store_true', help='Use FCC&3GP traces')
    parser.add_argument('--to', action='store_true', help='Use Oboe traces')
    parser.add_argument('--t3g', action='store_true', help='Use 3GP traces')
    parser.add_argument('--tp', action='store_true', help='Use Puffer traces')
    parser.add_argument('--tp2', action='store_true', help='Use Puffer2 traces')
    parser.add_argument('--tnf', action='store_true', help='Use Newfile traces')

    return parser.parse_args()


def calculate_from_selection(selected ,last_bit_rate):
    # selected_action is 0-7
    # naive step implementation
    if selected == 1:
        bit_rate = last_bit_rate
    elif selected == 2:
        bit_rate = last_bit_rate + 1
    else:
        bit_rate = last_bit_rate - 1
    # bound
    if bit_rate < 0:
        bit_rate = 0
    if bit_rate > 10:
        bit_rate = 10

    # print(bit_rate)
    return bit_rate


def main():
    args = parse_args()
    nn_model = NN_MODEL
    video = 'Mao'
    video_size_file = '../envs/video_size/' + video + '/video_size_'

    np.random.seed(RANDOM_SEED)


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


    all_cooked_time ,all_cooked_bw ,all_file_names = load_trace.load_trace(test_traces)

    # print(len(all_cooked_time[-1]))

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                                all_cooked_bw=all_cooked_bw, all_file_names = all_file_names,
                                video_size_file = video_size_file)

    # determine the QoE metric \
    REBUF_PENALTY = REBUF_PENALTY_log if args.log else REBUF_PENALTY_lin

    net_env.set_env_info(0, 0, 0, int(CHUNK_TIL_VIDEO_END_CAP), \
                          VIDEO_BIT_RATE, 1, REBUF_PENALTY, SMOOTH_PENALTY, 0)


    log_path = log_file_init + '_' + all_file_names[net_env.trace_idx]
    log_file = open( log_path ,'wb' )

    _, _, _, total_chunk_num, \
        bitrate_versions, rebuffer_penalty, smooth_penalty = net_env.get_env_info()

    with tf.compat.v1.Session() as sess:

        actor = a3c.ActorNetwork( sess ,
                                  state_dim=[S_INFO ,S_LEN] ,action_dim=A_DIM ,
                                  learning_rate=ACTOR_LR_RATE

                                  )

        sess.run( tf.compat.v1.global_variables_initializer() )
        saver = tf.compat.v1.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if nn_model is not None:  # NN_MODEL is the path to file
            saver.restore( sess ,nn_model )
            print( "Testing model restored." )

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros( A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros( (S_INFO ,S_LEN) )]
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
                _ = net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smoothness
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

            r_batch.append( reward )

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
            if len( s_batch ) == 0:
                state = [np.zeros( (S_INFO ,args.S_LEN) )]
            else:
                state = np.array( s_batch[-1] ,copy=True )

            # dequeue history record
            state = np.roll( state ,-1 ,axis=1 )

            # this should be S_INFO number of terms
            state[0 ,-1] = VIDEO_BIT_RATE[bit_rate] / float( np.max( VIDEO_BIT_RATE ) )  # last quality
            state[1 ,-1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2 ,-1] = float( video_chunk_size ) / float( delay ) / M_IN_K  # kilo byte / ms
            state[3 ,-1] = float( delay ) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4 ,:BITRATE_DIM] = np.array(next_video_chunk_sizes ) / M_IN_K / M_IN_K  # mega byte
            state[5 ,-1] = np.minimum(video_chunk_remain ,CHUNK_TIL_VIDEO_END_CAP ) / float( CHUNK_TIL_VIDEO_END_CAP )

            action_prob = actor.predict( np.reshape( state ,(1, S_INFO, S_LEN) ) )
            action_cumsum = np.cumsum( action_prob )
            selection = (action_cumsum > np.random.randint(1 ,RAND_RANGE ) / float( RAND_RANGE )).argmax()
            bit_rate = calculate_from_selection( selection ,last_bit_rate )
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            s_batch.append( state )

            entropy_record.append( a3c.compute_entropy( action_prob[0] ) )

            if end_of_video:
                log_file.write( '\n'.encode() )
                log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros( A_DIM )
                action_vec[selection] = 1

                s_batch.append( np.zeros( (S_INFO ,S_LEN) ) )
                a_batch.append( action_vec )
                entropy_record = []

                print("video count", video_count)
                video_count += 1

                if video_count >= len( all_file_names ):
                    break

                log_path = log_file_init + '_' + all_file_names[net_env.trace_idx]
                log_file = open(log_path, 'wb')

        print( "Test Done" )


if __name__ == '__main__':
    main()
