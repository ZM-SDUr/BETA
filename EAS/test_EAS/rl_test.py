import os
import sys
import numpy as np
import tensorflow as tf
import load_trace
import a3c
import fixed_env as env
from memory_profiler import memory_usage
import time
import matplotlib.pyplot as plt

tf.config.set_visible_devices([], 'GPU')
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8
A_DIM = 8
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [200., 800., 2200., 5000., 10000.,18000.,32000.,50000.]
HD = [1.32,2.28,4.52,33.0,41.0, 78.8, 101.2, 130.0]
LOG =[0.0, 1.386, 2.397, 3.218, 3.912, 4.5, 5.075 ,5.52]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 50.
SMOOTH_PENALTY = 1.
DEFAULT_QUALITY = 0  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = './results/log_sim_rl'
TEST_TRACES = '../../PPO-SAC-DQN/trace_for_test/'
band_level = [6.88, 10.97, 15.43, 20.78, 33.96]
NN_MODELs = ['../results/result-1/results5/nn_model_ep_5000.ckpt',
             '../results/result-2/results5/nn_model_ep_4000.ckpt',
             '../results/result-3/results5/nn_model_ep_4000.ckpt',
             '../results/result-4/results5/nn_model_ep_4000.ckpt',
             '../results/result-5/results5/nn_model_ep_4000.ckpt']

def f():

    start_num = int(0)
    end_num = int (100)

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)
    all_cooked_time = all_cooked_time[start_num:end_num]
    all_cooked_bw = all_cooked_bw[start_num:end_num]
    all_file_names = all_file_names[start_num:end_num]

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    with tf.compat.v1.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()  # save neural net parameters

        avg_bandwidth = np.mean(all_cooked_bw[net_env.trace_idx])
        closest_band_level = min(band_level, key=lambda x: abs(x - avg_bandwidth))
        model_index = band_level.index(closest_band_level)
        selected_model = NN_MODELs[model_index]
        if selected_model is not None:
            saver.restore(sess, selected_model)
            #print("Testing model restored.",selected_model)

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []
        entropy_ = 0.5

        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate]/1000. \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate])/1000.

            r_batch.append(reward)

            last_bit_rate = bit_rate

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp / M_IN_K) + ' ' +
                           str(VIDEO_BIT_RATE[bit_rate]) + ' ' +
                           str(buffer_size) + ' ' +
                           str(rebuf) + ' ' +
                           str(video_chunk_size) + ' ' +
                           str(delay) + ' ' +
                           str(entropy_) + ' ' +
                           str(reward) + '\n')

            log_file.flush()

            # retrieve previous state
            if len(s_batch) == 0:
                state = np.zeros((S_INFO, S_LEN))
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
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            s_batch.append(state)
            entropy_ = a3c.compute_entropy(action_prob[0])
            entropy_record.append(entropy_)

            if end_of_video:
                log_file.write('\n')
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

                video_count += 1
                #print(video_count)
                if video_count >= len(all_file_names):
                    break

                avg_bandwidth = np.mean(all_cooked_bw[net_env.trace_idx])
                closest_band_level = min(band_level, key=lambda x: abs(x - avg_bandwidth))
                model_index = band_level.index(closest_band_level)
                selected_model = NN_MODELs[model_index]
                if selected_model is not None:
                    saver.restore(sess, selected_model)
                    #print("Testing model restored.", selected_model)

                log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
                log_file = open(log_path, 'w')


start_time = time.time()
mem_usage = memory_usage(proc=f, interval=0.1)
end_time = time.time()

plt.figure(figsize=(10, 6))
plt.plot(mem_usage, label='Memory Usage (MiB)')

plt.title('Memory Usage Over Time')
print(end_time-start_time)
plt.plot(mem_usage)
print(np.mean(mem_usage),'MiB')

plt.show()


