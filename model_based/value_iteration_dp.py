from data.data import *


def get_optimal_q_value(data):
    q_value = np.zeros((data.n_states, data.n_actions))
    for r in range(data.max_iterations):
        prev_value = q_value

        # calculate value for states and actions
        q_value = data.reward_table + data.gamma * np.dot(data.transition_table, get_value(q_value))

        # convergence check
        if np.sum(q_value - prev_value) < 1e-3:
            print('converged!')
            break

    return q_value


def get_policy(q_value):
    # check which action gives best expected and assign it
    return np.argmax(q_value, axis=1)


def get_value(q_value):
    # check which action gives best expected and return it
    return np.max(q_value, axis=1)


def get_optimal_policy(data):
    # get max value function
    q_value = get_optimal_q_value(data)

    # improve the policy
    policy = get_policy(q_value)

    return policy, get_value(q_value)


if __name__ == '__main__':
    policy, value = get_optimal_policy(Data(n_states=5, n_actions=2))
    print('Optimal Value:', value)
    print('Optimal Policy:', policy)
