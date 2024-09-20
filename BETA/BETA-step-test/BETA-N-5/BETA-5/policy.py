import Gloabl_variable as gb
import tensorflow as tf
import numpy as np


def policy(state):
    sampled_actions = tf.squeeze(gb.actor_model(state))
    raw_action = sampled_actions
    sampled_actions = sampled_actions*(gb.upper_bound - gb.lower_bound) / 2. \
                      + (gb.upper_bound-gb.lower_bound) / 2. \
                      + gb.lower_bound

    legal_action = np.clip(sampled_actions.numpy(), gb.lower_bound, gb.upper_bound)
    sampled_actions = legal_action
    return np.squeeze(sampled_actions), np.squeeze(raw_action)
