from common_utils import *
from environment import *


def get_optimal_q_value(environment):
    q_value = np.zeros((environment.n_states, environment.n_actions))
    for r in range(environment.max_iterations):
        prev_value = q_value

        # calculate value for states and actions
        q_value = environment.reward_table + environment.gamma * np.dot(environment.transition_table,
                                                                        get_value_function(environment, greedy_policy(q_value), q_value))

        # convergence check
        if np.sum(q_value - prev_value) < 1e-3:
            print('converged!')
            break

    return q_value


def value_iteration(environment):
    # get max value function
    q_value = get_optimal_q_value(environment)

    # improve the policy
    policy = optimal_policy(q_value, greedy=True)

    return policy, get_value_function(environment, policy, q_value)


def main():
    environment = Environment(n_states=5, n_actions=2)
    policy, value = value_iteration(environment)
    print('Optimal Value:', value)
    print('Optimal Policy:', policy)


if __name__ == '__main__':
    main()
