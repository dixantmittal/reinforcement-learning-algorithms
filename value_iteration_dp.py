from data.data import *


def get_max_value():
    value = np.zeros(n_states)
    for r in range(n_iterations):
        prev_value = np.array(value)

        # calculate value for states maxed over actions
        value = np.max(data.gamma * np.dot(data.transition_table, value) + data.reward_table, axis=1)
        # convergence check
        if np.sum(value - prev_value) < 1e-3:
            print('converged!')
            break

    return value


def get_policy(value):
    # check which action gives best expected and assign it
    policy = np.argmax(data.gamma * np.dot(data.transition_table, value) + data.reward_table, axis=1)

    return policy


def main():
    # get max value function
    value = get_max_value()

    print('Optimal Value:', value)

    # improve the policy
    policy = get_policy(value)

    print('Optimal Policy:', policy)


if __name__ == '__main__':
    n_states = 16
    n_actions = 4
    n_iterations = 100

    data = Data(n_states, n_actions)

    main()
