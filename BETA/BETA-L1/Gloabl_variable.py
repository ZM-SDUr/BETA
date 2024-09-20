import nural_network as nn
import tensorflow as tf
import buffer as bf
import os
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created.")
    else:
        print(f"Directory {directory_path} already exists.")



num_states = 5
state_len = 8
num_actions = 1

policy_noise = 1.0

forward_step = 10

gamma = 0.99
tau = 0.005
########################################
actor_lr =  0.0001
critic_lr = 0.001

lower_bound =  200
upper_bound = 50000

save_log_count = 10000
save_model_timelog = 1

iteration_c = 4
total_episodes = 2000000

########################################
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
BIT_IN_BYTE = 8
REBUF_PENALTY = 50.0  # 4.3# 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1

RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = './log_file_4G-2/'
MODEL_SAVED = './model-L15-2/'
create_directory_if_not_exists(LOG_FILE)
create_directory_if_not_exists(MODEL_SAVED)

new_model = 0
Model_num =21

#TRACES_TCP_test = './trace_for_test/'
#TRACES_TCP_test  = './OPT_and_Model_con/classified_trace/L'+str(5)+'/'#'./Selected_trace/raw/' #'./TCP_trace_process_after/' + trace_file_list[0] + '/'
TRACES_TCP_train = '../L14/'
#TRACES_TCP_Va    = './validation_set/'

if new_model == 1:
    actor_model = nn.get_actor()
    critic_model1 = nn.get_critic()
    critic_model2 = nn.get_critic()
    target_actor = nn.get_actor()
    target_critic1 = nn.get_critic()
    target_critic2 = nn.get_critic()
else:
    get_model = '../BETA-14/model-L15/'
    actor_model = tf.keras.models.load_model(get_model+'actor_model'+str(Model_num))
    critic_model1 = tf.keras.models.load_model(get_model+'critic1_model'+str(Model_num))
    critic_model2 = tf.keras.models.load_model(get_model+'critic2_model'+str(Model_num))

    target_actor = tf.keras.models.load_model(get_model+'target_actor'+str(Model_num))
    target_critic1 = tf.keras.models.load_model(get_model+'target1_critic'+str(Model_num))
    target_critic2 = tf.keras.models.load_model(get_model+'target2_critic'+str(Model_num))

buffer = bf.Buffer(100000, 32)

target_actor.set_weights(actor_model.get_weights())
target_critic1.set_weights(critic_model1.get_weights())
target_critic2.set_weights(critic_model2.get_weights())

critic_optimizer1 = tf.keras.optimizers.Adam(critic_lr)
critic_optimizer2 = tf.keras.optimizers.Adam(critic_lr)
critic_grad_norm = 1.

actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
actor_grad_norm = 1.

