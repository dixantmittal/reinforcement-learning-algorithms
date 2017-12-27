from data.data import *

# for syntax/representation, refer data/data.py

# global constants
n_states = 16
n_actions = 4
n_iterations = 300

data = Data(n_states, n_actions)


def policy_evaluation(policy, value):
    for r in range(n_iterations):
        prev_value = np.array(value)

        # calculate values for all actions
        value = data.gamma * np.dot(np.transpose(data.transition_table, axes=(1, 0, 2)),
                                    value) + data.reward_table.transpose()

        # calculate value for policy action
        value = value[range(n_states), policy]

        # convergence check
        if np.sum(value - prev_value) < 1:
            break

    return value


def policy_improvement(value):
    # check which action gives best expected and assign it
    policy = np.argmax(np.dot(np.transpose(data.transition_table, axes=(1, 0, 2)), value), axis=1)

    return policy


def main():
    # initialize random policy
    policy = np.ones(n_states, dtype=np.int32)

    # initialize value function
    value = np.zeros(n_states)

    # until convergence
    for i in range(n_iterations):
        # for checking convergence
        prev_policy = policy

        # print current policy
        print(policy)

        # evaluate current policy
        value = policy_evaluation(policy, value)

        # improve the policy
        policy = policy_improvement(value)

        # if converged, stop
        if np.all(policy == prev_policy):
            break


if __name__ == '__main__':
    main()
