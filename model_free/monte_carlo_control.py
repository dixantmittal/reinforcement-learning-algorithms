import numpy as np
from tqdm import tqdm

from environment import Environment
from common_utils import *


def monte_carlo_policy_evaluation(environment, policy, q_value=None, return_q_value=True, alpha=1e-2):
    if q_value is None:
        q_value = np.zeros((environment.n_states, environment.n_actions))

    for _ in tqdm(range(environment.n_episodes)):
        # select a random starting state and action -> Exploring starts
        state = random_state(environment)
        action = random_action(environment)

        # generate an episode and get the actual discounted reward
        actual_reward = generate_episode(environment, state, action, policy)

        # update Expected Value
        q_value[state, action] = q_value[state, action] + alpha * (actual_reward - q_value[state, action])

    if return_q_value:
        return q_value
    else:
        return get_value_function(environment, policy, q_value)


def get_optimal_policy(environment):
    # get max value function
    q_value = np.zeros((environment.n_states, environment.n_actions))

    # generate random stochastic policy to evaluate
    policy = random_policy(environment)

    for _ in range(100):
        prev_policy = policy

        # evaluate policy
        q_value = monte_carlo_policy_evaluation(environment, policy, q_value)

        # improve the policy
        policy = optimal_policy(q_value, greedy=False)

        # if converged, stop
        if np.all(greedy_policy(policy) == greedy_policy(prev_policy)):
            break

    return policy


def main():
    policy = get_optimal_policy(Environment(n_states=5, n_actions=2, n_episodes=5000, max_iterations=10, episode_length=100))
    print('Monte Carlo Optimal Policy: ', policy)


if __name__ == '__main__':
    main()
