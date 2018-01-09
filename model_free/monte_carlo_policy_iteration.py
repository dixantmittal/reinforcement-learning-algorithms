import numpy as np
from tqdm import tqdm

from data.data import Data


def simulator(data, state, action):
    next_state_distribution = data.transition_model(state, action)
    return np.random.choice(data.n_states, p=next_state_distribution)


def generate_episode(data, state, action, policy):
    actual_reward = 0
    for i in range(max_episode_length):
        actual_reward = actual_reward + data.gamma ** i * data.reward(state, action)

        state = simulator(data, state, action)
        action = policy[state]

        if data.gamma * data.reward(state, action) < 1e-3:
            break

    return actual_reward


def policy_evaluation(data, policy, q_value, alpha=0.01):
    for _ in tqdm(range(data.n_episodes)):
        # select a random starting state and action
        state = np.random.randint(0, data.n_states)
        action = np.random.randint(0, data.n_actions)

        # generate an episode and get the actual discounted reward
        actual_reward = generate_episode(data, state, action, policy)

        # update Expected Value
        q_value[state, action] = q_value[state, action] + alpha * (actual_reward - q_value[state, action])
    return q_value


def policy_improvement(q_value):
    # check which action gives best expected and assign it
    return np.argmax(q_value, axis=1)


def get_optimal_policy(data):
    # get max value function
    q_value = np.zeros((data.n_states, data.n_actions))

    # generate random policy to evaluate
    prev_policy = policy = np.random.randint(0, data.n_actions, data.n_states)

    for _ in range(100):
        # evaluate policy
        q_value = policy_evaluation(data, policy, q_value)

        # improve the policy
        policy = policy_improvement(q_value)

        # if converged, stop
        if np.all(policy == prev_policy):
            break

        prev_policy = policy

    return policy


if __name__ == '__main__':
    max_episode_length = 100
    policy = get_optimal_policy(Data(n_states=5, n_actions=2, n_episodes=5000, max_iterations=10))
    print('Monte Carlo Optimal Policy: ', policy)
