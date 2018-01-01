import numpy as np
from tqdm import tqdm

from data.data import Data
from model_based.policy_iteration_dp import policy_evaluation as pe


def simulator(data, state, action):
    next_state_distribution = data.transition_model(state, action)
    return np.random.choice(data.n_states, p=next_state_distribution)


def policy_evaluation(data, policy, value, alpha=0.01):
    for _ in tqdm(range(data.n_episodes)):
        # select a random starting state
        state = np.random.randint(0, data.n_states)

        # select a random action and get reward
        action = policy[state]

        # get actual reward for taking this action
        actual_reward = data.reward(state, action)

        # transition to a new state in following this action
        next_state = simulator(data, state, action)

        # update Expected Value
        value[state] = value[state] + alpha * (actual_reward + data.gamma * value[next_state] - value[state])
    return value


def main(data):
    # generate zero value function
    value = np.zeros(data.n_states)

    # generate random policy to evaluate
    policy = np.random.randint(0, data.n_actions, data.n_states)

    # evaluate policy
    value = policy_evaluation(data, policy, value)

    print(pe(data, policy, value))
    print()
    print(value)


if __name__ == '__main__':
    main(Data(n_states=5, n_actions=2, n_episodes=20000))
