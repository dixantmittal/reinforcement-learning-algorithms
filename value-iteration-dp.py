from data.data import *

# for syntax/representation, refer data/data.py

# global constants
n_states = 16
n_iterations = 100


def get_max_value():
    value = np.zeros(n_states)
    for r in range(n_iterations):
        prev_value = np.array(value)

        # calculate value for states maxed over actions
        value = np.max(np.dot(np.transpose(transition_table, axes=(1, 0, 2)), value) + reward.reshape((-1, 1)), axis=1)

        # convergence check
        if np.sum(value - prev_value) < 1:
            break

    return value


def get_policy(value):
    # check which action gives best expected and assign it
    policy = np.argmax(np.dot(np.transpose(transition_table, axes=(1, 0, 2)), value), axis=1)

    return policy


def main():
    # get max value function
    value = get_max_value()

    # improve the policy
    policy = get_policy(value)

    print(policy)


if __name__ == '__main__':
    main()
