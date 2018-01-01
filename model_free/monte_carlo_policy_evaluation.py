import numpy as np
from tqdm import tqdm

from data.data import Data


def simulator(data, state, action):
    next_state_distribution = data.transition_model(state, action)
    return np.random.choice(data.n_states, p=next_state_distribution)


def generate_episode(data, state, policy):
    actual_reward = 0
    for i in range(max_episode_length):
        action = policy[state]
        actual_reward = actual_reward + data.gamma ** i * data.reward(state, action)

        state = simulator(data, state, action)

        if data.gamma * data.reward(state, action) < 1e-3:
            break

    return actual_reward


def policy_evaluation(data, policy, value, alpha=0.01):
    n_visits = np.zeros(data.n_states)
    for r in tqdm(range(data.n_episodes)):
        # select a random starting state
        state = np.random.randint(0, data.n_states)

        # generate an episode and get the actual discounted reward
        actual_reward = generate_episode(data, state, policy)

        # update Expected Value
        value[state] = value[state] + alpha * (actual_reward - value[state])
    return value


def main(data):
    # get max value function
    value = np.zeros(data.n_states)

    # generate random policy to evaluate
    policy = np.random.randint(0, data.n_actions, data.n_states)

    # evaluate policy
    value = policy_evaluation(data, policy, value)

    print(value)


if __name__ == '__main__':
    max_episode_length = 100
    main(Data(n_states=5, n_actions=2))
