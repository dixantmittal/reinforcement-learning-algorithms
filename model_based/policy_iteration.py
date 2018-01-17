from environment import *
from common_utils import *


def policy_evaluation(environment, policy, q_value=None, return_q_value=True):
    # generate zero value function
    if q_value is None:
        q_value = np.zeros((environment.n_states, environment.n_actions))

    for r in range(environment.max_iterations):
        prev_value = q_value

        # calculate values for all actions
        q_value = environment.reward_table + environment.gamma * np.dot(
            environment.transition_table, get_value_function(environment, policy, q_value))

        # convergence check
        if np.sum(q_value - prev_value) < 1:
            break

    if return_q_value:
        return q_value
    else:
        return get_value_function(environment, policy, q_value)


def policy_iteration(environment):
    # initialize random deterministic policy
    policy = np.random.randint(0, environment.n_actions, environment.n_states)

    # initialize value function
    q_value = np.zeros((environment.n_states, environment.n_actions))

    # until convergence
    for i in range(environment.max_iterations):
        # for checking convergence
        prev_policy = policy

        # evaluate current policy
        q_value = policy_evaluation(environment, policy)

        # improve the policy
        policy = optimal_policy(q_value, greedy=True)

        # if converged, stop
        if np.all(policy == prev_policy):
            break

    return policy, get_value_function(environment, policy, q_value)


if __name__ == '__main__':
    policy, value = policy_iteration(Environment(n_states=5, n_actions=2))
    print('Optimal Value:', value)
    print('Optimal Policy:', policy)
