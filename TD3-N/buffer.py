import tensorflow as tf
import numpy as np
import Gloabl_variable as gb

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=32):

        self.buffer_capacity = buffer_capacity

        self.batch_size = batch_size

        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, gb.num_states, gb.state_len))
        self.action_buffer = np.zeros((self.buffer_capacity, gb.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, gb.num_states, gb.state_len))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))


    def record(self, obs_tuple):

        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1

    @tf.function
    def update(
            self, state_batch, action_batch, reward_batch, next_state_batch,done_batch
    ):
        for it in range(gb.iteration_c):
            target_actions = gb.target_actor(next_state_batch, training=True)
            noise = tf.random.normal(shape=tf.shape(target_actions), mean=target_actions, stddev=gb.policy_noise)
            target_actions = tf.clip_by_value(noise, -1., 1.)
            target_q1 = gb.target_critic1([next_state_batch, target_actions], training=True)
            target_q2 = gb.target_critic2([next_state_batch, target_actions], training=True)
            target_Q = (target_q1+ target_q2)/2.0#tf.minimum()
            target_Q = reward_batch + (1-done_batch) *  (0.99**gb.forward_step) * target_Q

            with tf.GradientTape() as tape1:
                # 计算当前rewardg
                critic_value1 = gb.critic_model1([state_batch, action_batch], training=True)
                critic_loss1 = tf.math.reduce_mean(tf.math.square(target_Q - critic_value1))

            with tf.GradientTape() as tape2:
                critic_value2 = gb.critic_model2([state_batch, action_batch], training=True)
                critic_loss2 = tf.math.reduce_mean(tf.math.square(target_Q - critic_value2))

            critic_grad1 = tape1.gradient(critic_loss1, gb.critic_model1.trainable_variables)
            critic_grad1, _ = tf.clip_by_global_norm(critic_grad1, gb.actor_grad_norm)
            gb.critic_optimizer1.apply_gradients(zip(critic_grad1, gb.critic_model1.trainable_variables))

            critic_grad2 = tape2.gradient(critic_loss2, gb.critic_model2.trainable_variables)
            critic_grad2, _ = tf.clip_by_global_norm(critic_grad2, gb.actor_grad_norm)
            gb.critic_optimizer2.apply_gradients(zip(critic_grad2, gb.critic_model2.trainable_variables))

            if it >= gb.iteration_c - 1:
                with tf.GradientTape() as tape:
                    actions = gb.actor_model(state_batch, training=True)

                    critic_value = gb.critic_model1([state_batch, actions], training=True)

                    actor_loss = -tf.math.reduce_mean(critic_value)

                actor_grad = tape.gradient(actor_loss, gb.actor_model.trainable_variables)
                actor_grad, _ = tf.clip_by_global_norm(actor_grad, gb.actor_grad_norm)
                gb.actor_optimizer.apply_gradients(
                    zip(actor_grad, gb.actor_model.trainable_variables)
                )

    def learn(self, forward_n):

        record_range = min(self.buffer_counter - forward_n - 1, self.buffer_capacity - forward_n - 1)
        # Randomly sample indices
        start_indices = np.random.choice(record_range, self.batch_size)
        batch_indices = np.array([np.arange(i, i + forward_n) for i in start_indices])

        array_done = np.array(self.done_buffer)[batch_indices]
        array_reward = np.array(self.reward_buffer)[batch_indices]

        total_rewards = []
        for rewards_row, dones_row in zip(array_reward, array_done):
            total_reward = 0.0
            for reward, done in zip(reversed(rewards_row), reversed(dones_row)):
                total_reward = reward + gb.gamma * total_reward
                if done:
                    total_reward = reward
            total_rewards.append(total_reward)

        raw_done_batch = np.sum(array_done, axis=1)

        state_batch = tf.convert_to_tensor(self.state_buffer[start_indices])

        action_batch = tf.convert_to_tensor(self.action_buffer[start_indices])

        reward_batch = tf.convert_to_tensor(total_rewards)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)

        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[start_indices + forward_n])

        done_batch = tf.convert_to_tensor(raw_done_batch)
        done_batch = tf.cast(done_batch, dtype=tf.float32)

        self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)


    def del_buffer(self, del_number):
        self.state_buffer = self.state_buffer[del_number:]
        self.action_buffer = self.action_buffer[del_number:]
        self.reward_buffer = self.reward_buffer[del_number:]
        self.next_state_buffer = self.next_state_buffer[del_number:]
        self.done_buffer = self.done_buffer[del_number:]
        self.buffer_counter -= del_number

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))
