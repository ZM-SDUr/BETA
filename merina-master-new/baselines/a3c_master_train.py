import os

import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tflearn

EPSILON = 0.2
GAMMA = 0.99
ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6
ACTION_EPS = 1e-4
max_grad_norm = 10
H_target = 0.1

class ActorNetwork(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, bitrate_dim, entropy_weight, name = 'actor'):
        self.sess = sess
        self.s_dim = state_dim     # 状态维度，即环境状态的维度
        self.a_dim = action_dim    # 动作维度，即可能的动作数
        self.lr_rate = learning_rate
        self.bitrate_dim = bitrate_dim   # 比特率维度，
        self.scope_name = name

        self._entropy_weight = entropy_weight


        # Create the actor network
        self.inputs, self.out = self.create_actor_network()  # 定义输入和输出
        self.master_inputs, self.master_out = self.create_actor_network()  # 使用相同的网络结构，复用参数
        # self.master_inputs, self.master_out = self.create_actor_network()  # 定义输入和输出
        # 当前模型和旧模型的输出概率
        self.pro_old_tensor = tf.compat.v1.placeholder(tf.float32, [None, self.a_dim])
        # Get all network parameters
        self.network_params = \
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope = self.scope_name)

        # Set all network parameters
        self.input_network_params = []    # 初始化一个列表来存储网络参数的占位符
        for param in self.network_params:
            self.input_network_params.append(
                tf.compat.v1.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []    # 初始化一个操作列表，用于更新网络参数
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))
        # 选择的动作，一个0-1向量
        self.acts = tf.compat.v1.placeholder(tf.float32, [None, self.a_dim])
        self.master_acts = tf.compat.v1.placeholder(tf.float32, [None, self.a_dim])


        # 广义优势估计的结果
        self.Gae = tf.compat.v1.placeholder(tf.float32, [None, 1])    # 动作梯度权重的占位符
        # 普通计算的优势估计
        self.Adv = tf.compat.v1.placeholder(tf.float32, [None, 1])  # 动作梯度权重的占位符

        #self.maste = tf.compat.v1.placeholder(tf.float32)  # 存储ppo的损失
        self.entropy_weight = tf.compat.v1.placeholder(tf.float32)       # 熵权重的占位
        self.obj_1 = tf.compat.v1.placeholder(tf.float32)  # 熵权重的占位

        self.real_out = tf.clip_by_value(self.out, ACTION_EPS, 1. - ACTION_EPS)

        self.pro_old_log = tf.reduce_sum(tf.multiply(self.pro_old_tensor, self.acts),
                                        axis=1, keepdims=True)
        self.pro_log = tf.reduce_sum(tf.multiply(self.real_out, self.acts), axis=1,
                                    keepdims=True)

        self.master_pro = tf.reduce_sum(tf.multiply(self.master_out, self.master_acts), axis=1,
                                     keepdims=True)

        self.entropy = -tf.reduce_sum(tf.multiply(self.real_out, tf.compat.v1.log(self.real_out)), axis=1,
                                      keepdims=True)
        # 计算比例
        self.ratio = self.pro_log / self.pro_old_log
        # 计算裁剪后的目标函数
        self.clipped_ratios = tf.clip_by_value(self.ratio, 1. - EPSILON, 1. + EPSILON)
        # 计算最小优势函数
        self.ppo_loss = tf.minimum(self.ratio * self.Gae, self.clipped_ratios * self.Gae)
        self.total_loss = tf.where(tf.less(self.Gae, 0), tf.maximum(self.ppo_loss, 3. * self.Gae),self.ppo_loss)
        self.master_loss =tf.reduce_max(self.Gae) * 0.5 * self.master_pro

        self.obj =  -tf.reduce_sum(self.total_loss) - self.entropy_weight * tf.reduce_sum(self.entropy) - tf.reduce_sum(self.master_loss)
        # Optimization Op
        self.actor_optimize = tf.compat.v1.train.AdamOptimizer(self.lr_rate).minimize(self.obj)
    def create_actor_network(self):
        with tf.compat.v1.variable_scope(self.scope_name,reuse=tf.compat.v1.AUTO_REUSE):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])

            split_0 = tflearn.fully_connected(inputs[:, 0:1, -1], 128, activation='relu')
            split_1 = tflearn.fully_connected(inputs[:, 1:2, -1], 128, activation='relu')
            split_2 = tflearn.conv_1d(inputs[:, 2:3, :], 128, 4, activation='relu')
            split_3 = tflearn.conv_1d(inputs[:, 3:4, :], 128, 4, activation='relu')
            split_4 = tflearn.conv_1d(inputs[:, 4:5, :self.bitrate_dim], 128, 4, activation='relu')
            split_5 = tflearn.fully_connected(inputs[:, 5:6, -1], 128, activation='relu')

            # 将卷积层的输出扁平化
            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            # 合并不同的网络层输出
            merge_net = tflearn.merge([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')

            # 在合并后的网络上应用更多的全连接层
            dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')
            out = tflearn.fully_connected(dense_net_0, self.a_dim, activation='softmax')

            return inputs, out


    def train_2(self, inputs,pro_old, acts, Gae, Adv, master_s, master_a):

        self.sess.run(self.actor_optimize, feed_dict={
            self.inputs: inputs,
            self.pro_old_tensor: pro_old,
            self.master_inputs:master_s,
            self.master_acts:master_a,
            self.acts: acts,
            self.Gae: Gae,
            self.Adv: Adv,
            self.entropy_weight: self._entropy_weight,
        })

        en_weight = self.lr_rate
        pro_old = np.clip(pro_old, ACTION_EPS, 1. - ACTION_EPS)
        _H = np.mean(np.sum(-np.log(pro_old) * pro_old, axis=1))
        _g = _H - H_target
        self._entropy_weight -= en_weight * _g * 0.1




    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_gradients(self, inputs,pro_old, acts, Gae, entropy_weight):
        return self.sess.run(self.actor_optimize, feed_dict={
            self.inputs: inputs,
            self.pro_old_tensor:pro_old,
            self.acts:acts,
            self.Gae: Gae,
            self.entropy_weight : self._entropy_weight
        })
        pro_old = np.clip(pro_old, ACTION_EPS, 1. - ACTION_EPS)
        _H = np.mean(np.sum(-np.log(pro_old) * pro_old, axis=1))
        _g = _H - H_target
        self._entropy_weight -= self.lr_rate * _g * 0.1

    def apply_gradients(self, actor_gradients, learning_rate):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients+[self.lr_rate],  actor_gradients+[learning_rate])
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, state_dim, learning_rate, bitrate_dim, name = 'critic'):
        self.sess = sess
        self.s_dim = state_dim
        self.lr_rate = learning_rate
        self.bitrate_dim = bitrate_dim
        self.scope_name = name

        # Create the critic network
        self.inputs, self.out = self.create_critic_network()

        # Get all network parameters
        self.network_params = \
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf.placeholder(tf.float32, [None, 1])


        # Mean square error
        self.loss = tflearn.mean_square(self.td_target, self.out)

        # Compute critic gradient
        #self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        self.critic_optimize = tf.train.AdamOptimizer(self.lr_rate).minimize(self.loss)

    def create_critic_network(self):
        with tf.compat.v1.variable_scope(self.scope_name):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])

            split_0 = tflearn.fully_connected(inputs[:, 0:1, -1], 128, activation='relu')
            split_1 = tflearn.fully_connected(inputs[:, 1:2, -1], 128, activation='relu')
            split_2 = tflearn.conv_1d(inputs[:, 2:3, :], 128, 4, activation='relu')
            split_3 = tflearn.conv_1d(inputs[:, 3:4, :], 128, 4, activation='relu')
            split_4 = tflearn.conv_1d(inputs[:, 4:5, :self.bitrate_dim], 128, 4, activation='relu')
            # split_5 = tflearn.fully_connected(inputs[:, 4:5, -1], 128, activation='relu')
            split_5 = tflearn.fully_connected(inputs[:, 5:6, -1], 128, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            merge_net = tflearn.merge([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')

            dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')
            out = tflearn.fully_connected(dense_net_0, 1, activation='linear')

            return inputs, out

    def train(self, inputs, td_target):
        self.sess.run(self.critic_optimize, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_td(self, inputs, td_target):
        return self.sess.run(self.td, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def get_gradients(self, inputs, td_target):
        return self.sess.run(self.critic_optimize, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def apply_gradients(self, critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


def compute_gradients_actor(s_batch, pro_old,  acts, Gae, Adv, actor, entropy_weight):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """


    actor.train(s_batch, pro_old, acts,Gae, Adv, entropy_weight)



def compute_gradients_critic(s_batch, R_batch, critic):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """

    critic.train(s_batch, R_batch)





def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out


def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H


def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)
    avg_entropy = tf.Variable(0.)
    tf.summary.scalar("Avg_entropy", avg_entropy)

    summary_vars = [td_loss, eps_total_reward, avg_entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
