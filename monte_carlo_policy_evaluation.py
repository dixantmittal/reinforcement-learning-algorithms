import numpy as np
from tqdm import tqdm

from data.data import Data

n_states = 5
n_actions = 2
n_episodes = 10000
max_episode_length = 100

data = Data(n_states, n_actions)


def simulator(state, action):
    next_state_distribution = data.transition_model(state, action)
    return np.random.choice(n_states, p=next_state_distribution)


def generate_episode(state, policy):
    actual_reward = 0
    for i in range(max_episode_length):
        action = policy[state]
        actual_reward = actual_reward + data.gamma ** i * data.reward(state, action)

        state = simulator(state, action)

        if data.gamma * data.reward(state, action) < 1e-3:
            break

    return actual_reward


def policy_evaluation(policy, value):
    n_visits = np.zeros(n_states)
    for r in tqdm(range(n_episodes)):
        # select a random starting state
        state = np.random.randint(0, n_states)

        # generate an episode and get the actual discounted reward
        actual_reward = generate_episode(state, policy)

        # update Expected Value
        value[state] = value[state] * n_visits[state] + actual_reward
        n_visits[state] += 1
        value[state] /= n_visits[state]
    return value


def main():
    # get max value function
    value = np.zeros(n_states)

    # generate random policy to evaluate
    policy = np.random.randint(0, n_actions, n_states)

    # evaluate policy
    value = policy_evaluation(policy, value)

    print(value)


if __name__ == '__main__':
    main()
