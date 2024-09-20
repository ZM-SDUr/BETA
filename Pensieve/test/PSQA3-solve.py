import numpy as np
import fixed_env as env
import load_trace
import copy
import sys
import itertools
from memory_profiler import memory_usage
import time
import matplotlib.pyplot as plt

S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 8
MPC_FUTURE_CHUNK_COUNT = 4
VIDEO_BIT_RATE = [200, 800, 2200, 5000, 10000,18000,32000,50000]
HD = [1.32,2.28,4.52,33.0,41.0, 78.8, 101.2, 130]
LOG= [0.0, 1.386, 2.397, 3.218, 3.912, 4.5, 5.075 ,5.52]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
TAU = 60.0
Gamma  = 0.8
START_point = 0
M_IN_K = 1000.0
REBUF_PENALTY = 5.52
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 0  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
SUMMARY_DIR = './PSQA_result-LOG/'
LOG_FILE = './PSQA_result-LOG/PSQA'

pre_len = 7
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/nn_model_ep_5900.ckpt'




def get_20_reward(env_trace_id, gamma_sub):
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace('../../PPO-SAC-DQN/trace_for_test/')

    net_env_sub = env.Environment(all_cooked_time=all_cooked_time[env_trace_id:],
                              all_cooked_bw=all_cooked_bw[env_trace_id:])
    net_env_sub.trace_idx = env_trace_id

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []


    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real

        delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = net_env_sub.get_video_chunk(
            bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[0]))
        log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))
        reward = log_bit_rate - REBUF_PENALTY * rebuf - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

        r_batch.append(reward)

        last_bit_rate = bit_rate


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

        past_bandwidths = state[3, -5:]

        while past_bandwidths[0] == 0.0:
            if len(past_bandwidths) <= 1:
                break
            past_bandwidths = past_bandwidths[1:]

        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1 / float(past_val + 0.001))
        harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))

        harmonic_bandwidth = harmonic_bandwidth * 8000

        bit_rate = gamma_sub * harmonic_bandwidth * (buffer_size + TAU) / TAU

        min_difference = 10000000

        curr_bitrate = bit_rate
        for index, target_bitrate in enumerate(VIDEO_BIT_RATE):
            difference = abs(curr_bitrate - target_bitrate)
            if difference < min_difference:
                min_difference = difference
                bit_rate = index

        s_batch.append(state)

        if len(r_batch)==pre_len:
            totalr = np.sum(r_batch)

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]
            break


    return totalr


def f():

    start_num = int(0)
    end_num = int (100)


    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace('../../PPO-SAC-DQN/trace_for_test/')
    all_cooked_time = all_cooked_time[start_num:end_num]
    all_cooked_bw = all_cooked_bw[start_num:end_num]
    all_file_names = all_file_names[start_num:end_num]
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)
    net_env.trace_idx = START_point
    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []

    video_count = 0



    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay,sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        #reward = LOG[bit_rate] \
        #         - REBUF_PENALTY * rebuf \
        #         - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
        #                                   VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[0]))
        log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))
        reward = log_bit_rate - REBUF_PENALTY * rebuf - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

        r_batch.append(reward)
        last_bit_rate = bit_rate

        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\n')
        log_file.flush()

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

        past_bandwidths = state[3, -5:]

        while past_bandwidths[0] == 0.0:
            if len(past_bandwidths) <= 1:
                break
            past_bandwidths = past_bandwidths[1:]

        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1/float(past_val+0.001))
        harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

        harmonic_bandwidth = harmonic_bandwidth * 8000

        mem_gamma = Gamma
        if len(r_batch)==pre_len:
            mem_reward = -20000
            for gamma_s in [0.4, ]:#0.2,  0.4,  0.6, 0.8,  1.0,  1.2,  1.4, 1.6 ,1.8
                get_r  =  get_20_reward(net_env.trace_idx , gamma_s)
                if get_r> mem_reward:
                    mem_reward = get_r
                    mem_gamma = gamma_s
            print(net_env.trace_idx,mem_gamma)


        bit_rate = mem_gamma * harmonic_bandwidth * (buffer_size + TAU) / TAU

        min_difference = 10000000




        curr_bitrate = bit_rate
        for index, target_bitrate in enumerate(VIDEO_BIT_RATE):
            difference = abs(curr_bitrate - target_bitrate)
            if difference < min_difference:
                min_difference = difference
                bit_rate = index

        s_batch.append(state)

        if end_of_video:

            env_mem = copy.deepcopy(net_env)
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            print("QoE:",np.sum(r_batch))
            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            entropy_record = []

            print("video count: ", video_count," file: ",all_file_names[net_env.trace_idx])
            video_count += 1

            if video_count >= len(all_file_names):
            #if video_count >= 1000:
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')

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