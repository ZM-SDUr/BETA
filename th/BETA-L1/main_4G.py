
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

'''
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[5], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[5], True)
'''
import env
import load_trace
import numpy as np
import policy as po
import buffer as bf
import Gloabl_variable as gb

plt_len = gb.save_log_count
reward_list = np.zeros((plt_len, 1))
action_list = np.zeros((plt_len, 1))
rebuf_list = np.zeros((plt_len, 1))
throughput_list = np.zeros((plt_len, 1))
critic_list = np.zeros((plt_len, 1))
time_list = np.zeros((plt_len, 1))
buffer_size_list = np.zeros((plt_len, 1))

all_cooked_time, all_cooked_bw, all_file_name = load_trace.load_trace(gb.TRACES_TCP_train)

net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw)

time_stamp = 0
pre_state = np.zeros((gb.num_states, gb.state_len))
state = pre_state
action_pre = 200
episodic_count = 0
episodic_reward = 0

video_level  = [200., 800., 2200., 5000., 10000.,18000.,32000.,50000.]
HD = [1.32,2.28,4.52,33.0,41.0, 78.8, 101.2, 130.0]
LOG =[0.0, 1.386, 2.397, 3.218, 3.912, 4.5, 5.075 ,5.52]

def find_closest(levels, target):
    # 从列表中找到最接近target的值和其索引
    closest_index, closest_value = min(enumerate(levels), key=lambda x: abs(x[1] - target))
    return closest_index, closest_value
video_chunk_remain = 48
while True:  # experience video streaming forever

    pre_state = tf.expand_dims(tf.convert_to_tensor(pre_state), 0)

    action, _ = po.policy(pre_state)
    if video_chunk_remain==48:
        action = 200

    action_index, action  = find_closest(video_level, action)

    raw_action = (action - gb.lower_bound - (gb.upper_bound - gb.lower_bound) / 2.) / (
                (gb.upper_bound - gb.lower_bound) / 2.)

    delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, end_of_video, video_chunk_remain = \
        net_env.get_video_chunk(action)
    print(episodic_count, action, '[',float(video_chunk_size) * gb.BIT_IN_BYTE / float(delay) / gb.M_IN_K,']', all_file_name[net_env.trace_idx],[buffer_size])

    time_stamp += delay  # in ms
    time_stamp += sleep_time  # in ms

    state = np.roll(state, -1, axis=1)
    # this should be S_INFO number of terms
    state[0, -1] = raw_action  # last quality
    state[1, -1] = buffer_size / gb.BUFFER_NORM_FACTOR # 10 sec
    state[2, -1] = float(video_chunk_size) / float(delay) / gb.M_IN_K    # kilo byte / ms
    state[3, -1] = float(delay) / gb.M_IN_K / gb.BUFFER_NORM_FACTOR  # 10 sec
    state[4, -1] = np.minimum(video_chunk_remain, gb.CHUNK_TIL_VIDEO_END_CAP) / float(gb.CHUNK_TIL_VIDEO_END_CAP)

    # -- linear reward --
    # reward is video quality - rebuffer penalty - smoothness
    #action / gb.M_IN_K

    reward = action/1000. \
             - gb.REBUF_PENALTY * rebuf \
             - gb.SMOOTH_PENALTY * np.abs(action - action_pre) / gb.M_IN_K

    episodic_reward += reward
    #if episodic_count%150!=0:
    gb.buffer.record([pre_state, raw_action, reward, state, end_of_video])
    # episodic_reward += reward

    if end_of_video:
        print("episodic_reward: " + str(episodic_reward))
        episodic_reward = 0
        state = np.zeros((gb.num_states, gb.state_len))
        raw_action = -1
        action = 200
        video_chunk_remain = 48
        pre_state = state
    action_pre = action

    reward_list[episodic_count % plt_len] = reward
    action_list[episodic_count % plt_len] = action
    rebuf_list[episodic_count % plt_len] = rebuf
    throughput_list[episodic_count % plt_len] = float(video_chunk_size) * gb.BIT_IN_BYTE / float(delay) / gb.M_IN_K
    buffer_size_list[episodic_count % plt_len] = buffer_size



    if episodic_count >  gb.forward_step + 5 :
        gb.buffer.learn(gb.forward_step)
        bf.update_target(gb.target_actor.variables, gb.actor_model.variables, gb.tau)
        bf.update_target(gb.target_critic1.variables, gb.critic_model1.variables, gb.tau)
        bf.update_target(gb.target_critic2.variables, gb.critic_model2.variables, gb.tau)

    episodic_count += 1

    if episodic_count % plt_len == 0:
        with open(gb.LOG_FILE + "recent_log" + str(int(episodic_count / plt_len)), "w") as f:
            for i in range(plt_len):
                f.write(str(action_list[i][0]) + " ")
                f.write(str(buffer_size_list[i][0]) + " ")
                f.write(str(reward_list[i][0]) + " ")
                f.write(str(throughput_list[i][0]) + " ")
                f.write("\n")

    if episodic_count % (gb.save_log_count * gb.save_model_timelog) == 0:
        gb.actor_model.save(
            gb.MODEL_SAVED + 'actor_model' + str(int(episodic_count / (gb.save_log_count * gb.save_model_timelog))),
            include_optimizer=True)
        gb.critic_model1.save(
            gb.MODEL_SAVED + 'critic1_model' + str(
                int(episodic_count / (gb.save_log_count * gb.save_model_timelog))),
            include_optimizer=True)
        gb.critic_model2.save(
            gb.MODEL_SAVED + 'critic2_model' + str(
                int(episodic_count / (gb.save_log_count * gb.save_model_timelog))),
            include_optimizer=True)

        gb.target_actor.save(
            gb.MODEL_SAVED + 'target_actor' + str(int(episodic_count / (gb.save_log_count * gb.save_model_timelog))),
            include_optimizer=True)
        gb.target_critic1.save(
            gb.MODEL_SAVED + 'target1_critic' + str(
                int(episodic_count / (gb.save_log_count * gb.save_model_timelog))),
            include_optimizer=True)
        gb.target_critic2.save(
            gb.MODEL_SAVED + 'target2_critic' + str(
                int(episodic_count / (gb.save_log_count * gb.save_model_timelog))),
            include_optimizer=True)
    if episodic_count >= gb.total_episodes:
        break
    pre_state = state
    action_pre = action
