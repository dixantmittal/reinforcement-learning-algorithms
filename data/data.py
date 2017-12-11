# Each state is treated as an index in the array. 's' Reward is written as 1D array assigning a reward value [R(s)]
# to each state. Transition probability table is represented as a 3D matrix with axis: (action, current state,
# next state) If there is a probability for a state s to transition to some other state s' under action a,
# probability value is given by table[a,s,s']

import numpy as np

states = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
actions = {0, 1, 2, 3, 4, 5}

reward = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

transition_table = np.load(open('data/transition-table.npy', 'rb'))


def transition_model(state, action=None):
    if action is None:
        return transition_table[:, state]
    else:
        return transition_table[action, state]
