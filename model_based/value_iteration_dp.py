from data.data import *


def get_max_value(data):
    value = np.zeros(data.n_states)
    for r in range(data.max_iterations):
        prev_value = np.array(value)

        # calculate value for states maxed over actions
        value = np.max(data.gamma * np.dot(data.transition_table, value) + data.reward_table, axis=1)
        # convergence check
        if np.sum(value - prev_value) < 1e-3:
            print('converged!')
            break

    return value


def get_policy(data, value):
    # check which action gives best expected and assign it
    policy = np.argmax(data.gamma * np.dot(data.transition_table, value) + data.reward_table, axis=1)

    return policy


def main(data):
    # get max value function
    value = get_max_value(data)

    print('Optimal Value:', value)

    # improve the policy
    policy = get_policy(data, value)

    print('Optimal Policy:', policy)


if __name__ == '__main__':
    main(Data(n_states=5, n_actions=2))
