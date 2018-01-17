import numpy as np
from tqdm import tqdm

from environment import Environment
from common_utils import *
from sarsa_policy_evaluation import sarsa_policy_evaluation as policy_evaluation


def td_control(environment):
    # generate random policy to evaluate
    policy = random_policy(environment)

    q_value = None

    for _ in range(environment.max_iterations):
        prev_policy = policy

        # evaluate policy
        q_value = policy_evaluation(environment, policy, return_q_value=True)

        # improve policy
        policy = optimal_policy(q_value, greedy=False)

        # if converged, stop
        if np.all(greedy_policy(policy) == greedy_policy(prev_policy)):
            break

    return policy, get_value_function(environment, policy, q_value)


def main():
    environment = Environment(n_states=20, n_actions=5, n_episodes=50000)

    policy, value = td_control(environment)
    print('TD Optimal Value:', value)
    print('TD Optimal Policy:', policy)


if __name__ == '__main__':
    main()
