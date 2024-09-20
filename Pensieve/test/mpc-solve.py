import numpy as np
import fixed_env as env
import load_trace
import itertools
import sys
import random
from memory_profiler import memory_usage
import time
import matplotlib.pyplot as plt

S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 8
MPC_FUTURE_CHUNK_COUNT = 4
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [200, 800, 2200, 5000, 10000,18000,32000,50000]
HD = [1.32,2.28,4.52,33.0,41.0, 78.8, 101.2, 130]
LOG= [0.0, 1.386, 2.397, 3.218, 3.912, 4.5, 5.075 ,5.52]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
M_IN_K = 1000.0
REBUF_PENALTY = 50.
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 0  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
LOG_FILE = './MPC_result/mpc'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/nn_model_ep_5900.ckpt'

coun_num=0
CHUNK_COMBO_OPTIONS = []

# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []

def get_chunk_size(quality, index):
    if ( index < 0 or index > 48 ):
        return 0
    # note that the quality and video labels are inverted (i.e., quality 4 is highest and this pertains to video1)
    #sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index], 1: size_video5[index], 0:size_video6[index]}
    chunk_size = VIDEO_BIT_RATE[quality]*500
    return chunk_size


def f():

    start_num = int(0)
    end_num = int (100)
    rows = 10e5
    cols = 6

    random_matrix = [[random.random() for _ in range(int(cols))] for _ in range(int(rows))]

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace('../../PPO-SAC-DQN/trace_for_test/')
    all_cooked_time = all_cooked_time[start_num:end_num]
    all_cooked_bw = all_cooked_bw[start_num:end_num]
    all_file_names = all_file_names[start_num:end_num]
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)


    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 0

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []

    video_count = 0



    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay,sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes,end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty



        reward = VIDEO_BIT_RATE[0]/1000. - REBUF_PENALTY * rebuf\
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[0] - VIDEO_BIT_RATE[0])/1000.


        r_batch.append(reward)


        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward


        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = rebuf
        state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

        # ================== MPC =========================
        curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if ( len(past_bandwidth_ests) > 0 ):
            curr_error  = abs(past_bandwidth_ests[-1]-state[3,-1])/float(state[3,-1])
        past_errors.append(curr_error)

        # pick bitrate according to MPC
        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[3,-5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]
        #if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        #else:
        #    past_bandwidths = state[3,-5:]
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1/float(past_val))
        harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        max_error = 0
        error_pos = -5
        if ( len(past_errors) < 5 ):
            error_pos = -len(past_errors)
        max_error = float(max(past_errors[error_pos:]))
        #future_bandwidth = 0.8*harmonic_bandwidth/(1.0+max_error)  # robustMPC here
        past_bandwidth_ests.append(harmonic_bandwidth)


        # future chunks length (try 4 if that many remaining)
        last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain)
        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if ( TOTAL_VIDEO_CHUNKS - last_index < 5 ):
            future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index

        column_index = coun_num#random.randint(0, cols - 1)

        selected_column = [row[column_index] for row in random_matrix]
        bit_rate = int(int(future_chunk_length *selected_column[0])%8)



        s_batch.append(state)

        if end_of_video:

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            print(np.sum(r_batch))
            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            entropy_record = []

            print("video count", video_count,' file: ',all_file_names[net_env.trace_idx])
            video_count += 1

            if video_count >= len(all_file_names):
                break




if __name__ == "__main__":
    start_time = time.time()
    mem_usage = memory_usage(proc=f, interval=0.1)
    end_time = time.time()

    plt.figure(figsize=(10, 6))
    plt.plot(mem_usage, label='Memory Usage (MiB)')

    plt.title('Memory Usage Over Time')
    print(end_time - start_time)
    plt.plot(mem_usage)
    print(np.mean(mem_usage), 'MiB')

    plt.show()
