import tensorflow as tf
import sys

tf.config.set_visible_devices([], 'GPU')

import load_trace
import numpy as np
import Gloabl_variable as gb
import fixed_env as env
import matplotlib.pyplot as plt

VIDEO_BIT_RATE = [200., 800., 2200., 5000., 10000., 18000., 32000., 50000.]

ALL_REWARD = []


def main(arg1, arg2, model_num):
    start_num = int(arg1)
    end_num = int(arg2)
    model_num = int(model_num)

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace('../BETA2/test-L5/')
    all_cooked_time = all_cooked_time[start_num:end_num]
    all_cooked_bw = all_cooked_bw[start_num:end_num]
    all_file_names = all_file_names[start_num:end_num]
    net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw)

    Model_reward_list = []
    get_model = tf.keras.models.load_model(gb.MODEL_SAVED+'actor_model' + str(model_num))

    trace_test_count = 0
    DEFAULT_BITRATE = 200
    action = DEFAULT_BITRATE
    action_pre = action
    pre_state = np.zeros((gb.num_states, gb.state_len))
    state = pre_state
    action_list = []
    reward_list = []
    rebuf_list = []
    buffer_size_list = []
    chunk_size_list = []
    delay_list = []

    while True:
        pre_state = tf.expand_dims(tf.convert_to_tensor(pre_state), 0)
        Action_model = tf.squeeze(get_model(pre_state))
        action_model = Action_model
        selected_action = np.squeeze(action_model.numpy())
        selected_action *= (gb.upper_bound - gb.lower_bound) / 2.
        selected_action += (gb.upper_bound - gb.lower_bound) / 2.
        selected_action += gb.lower_bound
        action = np.clip(selected_action, gb.lower_bound, gb.upper_bound)
        action = min(VIDEO_BIT_RATE, key=lambda x: abs(x - action))
        raw_action = (action - gb.lower_bound - (gb.upper_bound - gb.lower_bound) / 2.) / (
                (gb.upper_bound - gb.lower_bound) / 2.)

        delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, \
            end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(action)

        if video_chunk_remain >= 47:
            action = DEFAULT_BITRATE
        state = np.roll(state, -1, axis=1)
        state[0, -1] = raw_action  # last quality
        state[1, -1] = buffer_size / gb.BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / float(delay) / gb.M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / gb.M_IN_K / gb.BUFFER_NORM_FACTOR  # 10 sec
        state[4, -1] = np.minimum(video_chunk_remain, gb.CHUNK_TIL_VIDEO_END_CAP) / float(gb.CHUNK_TIL_VIDEO_END_CAP)

        reward = action / gb.M_IN_K \
                 - gb.REBUF_PENALTY * rebuf \
                 - gb.SMOOTH_PENALTY * np.abs(action - action_pre) / gb.M_IN_K

        action_list.append(action)
        reward_list.append(reward)
        rebuf_list.append(rebuf)
        buffer_size_list.append(buffer_size)
        chunk_size_list.append(video_chunk_size)
        delay_list.append(delay)
        #troughput_list.append(float(video_chunk_size) / float(delay) / gb.M_IN_K)

        action_pre = action
        pre_state = state

        if end_of_video:

            log_path = "../test_all/" + all_file_names[net_env.trace_idx - 1]
            with open(log_path, 'w') as f:
                for i in range(len(action_list)):
                    f.write(str(action_list[i]) + ' ')
                    f.write(str(reward_list[i]) + ' ')
                    f.write(str(rebuf_list[i]) + ' ')
                    f.write(str(buffer_size_list[i]) + ' ')
                    f.write(str(chunk_size_list[i]) + ' ')
                    f.write(str(delay_list[i]) + ' ')
                    #f.write(str(troughput_list[i]) + ' ')
                    f.write('\n')

            trace_reward = np.sum(reward_list[:])
            Model_reward_list.append(trace_reward)
            action_list = []
            reward_list = []
            rebuf_list = []
            buffer_size_list = []
            #troughput_list = []
            chunk_size_list = []
            delay_list = []

            action = DEFAULT_BITRATE
            action_pre = action
            pre_state = np.zeros((gb.num_states, gb.state_len))
            state = pre_state
            trace_test_count += 1

        if trace_test_count >= len(all_file_names):
            break
    ALL_REWARD.append(np.mean(Model_reward_list))
    print(f"Model {model_num} Average Reward: {np.mean(ALL_REWARD)}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: test_actor.py <arg1> <arg2> <model_num>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])

'''


'''