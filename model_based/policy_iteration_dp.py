from data.data import *


def policy_evaluation(data, policy, q_value):
    for r in range(data.max_iterations):
        prev_value = np.array(q_value)

        # calculate values for all actions
        q_value = data.reward_table + data.gamma * np.dot(data.transition_table, q_value[range(data.n_states), policy])

        # convergence check
        if np.sum(q_value - prev_value) < 1:
            break

    return q_value


def get_value(q_value):
    # check which action gives best expected and return it
    return np.max(q_value, axis=1)


def policy_improvement(data, q_value):
    # check which action gives best expected and assign it
    return np.argmax(q_value, axis=1)


def get_optimal_policy(data):
    # initialize random policy
    policy = np.random.randint(0, data.n_actions, data.n_states)

    # initialize value function
    q_value = np.zeros((data.n_states, data.n_actions))

    # until convergence
    for i in range(data.max_iterations):
        # for checking convergence
        prev_policy = policy

        # evaluate current policy
        q_value = policy_evaluation(data, policy, q_value)

        # improve the policy
        policy = policy_improvement(data, q_value)

        # if converged, stop
        if np.all(policy == prev_policy):
            print('converged!')
            break

    return policy, get_value(q_value)


if __name__ == '__main__':
    policy, value = get_optimal_policy(Data(n_states=5, n_actions=2))
    print('Optimal Value:', value)
    print('Optimal Policy:', policy)
