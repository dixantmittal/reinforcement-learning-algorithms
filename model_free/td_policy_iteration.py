import numpy as np
from tqdm import tqdm

from data.data import Data


def simulator(data, state, action):
    next_state_distribution = data.transition_model(state, action)
    return np.random.choice(data.n_states, p=next_state_distribution)


def get_value(q_value):
    # check which action gives best expected and return it
    return np.max(q_value, axis=1)


def policy_evaluation(data, policy, q_value, alpha=0.01):
    for _ in tqdm(range(data.n_episodes)):
        # select a random starting state
        state = np.random.randint(0, data.n_states)

        # select a random action and get reward
        action = np.random.randint(0, data.n_actions)

        # get actual reward for taking this action
        actual_reward = data.reward(state, action)

        # transition to a new state in following this action
        next_state = simulator(data, state, action)

        # update Expected Value
        q_value[state, action] = q_value[state, action] + alpha * (
                actual_reward + data.gamma * q_value[next_state, policy[next_state]] - q_value[state, action])
    return q_value


def policy_improvement(q_value):
    # check which action gives best expected and assign it
    return np.argmax(q_value, axis=1)


def get_optimal_policy(data):
    # generate zero value function
    q_value = np.zeros((data.n_states, data.n_actions))

    # generate random policy to evaluate
    prev_policy = policy = np.random.randint(0, data.n_actions, data.n_states)

    for _ in range(100):
        # evaluate policy
        q_value = policy_evaluation(data, policy, q_value)

        policy = policy_improvement(q_value)

        # if converged, stop
        if np.all(policy == prev_policy):
            break

        prev_policy = policy

    return policy, get_value(q_value)


if __name__ == '__main__':
    policy, value = get_optimal_policy(Data(n_states=5, n_actions=2, n_episodes=10000))
    print('TD Optimal Value:', value)
    print('TD Optimal Policy:', policy)
