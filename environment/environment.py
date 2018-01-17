# Each state is treated as an index in the array 's' .Reward is written as 2D array assigning a reward value [R(s,
# a)] to each state and corresponding action. Transition probability table is represented as a 3D matrix with axis: (
# action, current state, next state) If there is a probability for a state s to transition to some other state s'
# under action a, probability value is given by table[s,a,s']

import numpy as np


class Environment(object):

    def __init__(self, n_states, n_actions, max_iterations=100, n_episodes=1000, episode_length=100):
        table = np.random.rand(n_states, n_actions, n_states)
        mask = np.random.rand(n_states, n_actions, n_states) < 0.25
        table[mask] = 0
        for i in range(n_states):
            table[i, :, i] = table[i, :, i] + 1e-10
        table = table / (np.sum(table, axis=2, keepdims=True) + 1e-10)

        self.transition_table = table
        self.gamma = 0.9
        self.reward_table = np.random.randint(low=0, high=10, size=n_states * n_actions).reshape(n_states, n_actions)

        self.n_states, self.n_actions = n_states, n_actions
        self.max_iterations = max_iterations
        self.n_episodes = n_episodes
        self.episode_length = episode_length

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

    def simulate(self, state, action):
        next_state_distribution = self.transition_model(state, action)
        return self.reward(state, action), np.random.choice(self.n_states, p=next_state_distribution)
