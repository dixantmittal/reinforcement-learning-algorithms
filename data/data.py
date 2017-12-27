# Each state is treated as an index in the array 's' .Reward is written as 2D array assigning a reward value [R(s,a)]
# to each state and corresponding action. Transition probability table is represented as a 3D matrix with axis: (action, current state,
# next state) If there is a probability for a state s to transition to some other state s' under action a,
# probability value is given by table[s,a,s']

import numpy as np


class Data(object):

    def __init__(self, n_states, n_actions):
        table = np.random.rand(n_states, n_actions, n_states)
        mask = np.random.rand(n_states, n_actions, n_states) < 0.5
        table[mask] = 0
        table = table / (np.sum(table, axis=2, keepdims=True) + 1e-20)

        self.transition_table = table
        self.gamma = 0.9
        self.reward_table = np.random.randint(low=0, high=10, size=n_states * n_actions).reshape(n_states, n_actions)

    def transition_model(self, state, action=None):
        if action is None:
            return self.transition_table[state]
        else:
            return self.transition_table[state, action]

    def reward(self, state=None, action=None):

        if action is None:
            return self.reward_table[state]

        else:
            return self.reward_table[state, action]
