import numpy as np


def choose_action(environment, policy, state, greedy=False):
    if len(policy.shape) == 1:
        policy = convert_to_one_hot_vector(environment, policy)

    if greedy:
        return np.argmax(policy[state])
    else:
        return np.random.choice(environment.n_actions, p=policy[state])


def convert_to_one_hot_vector(environment, policy):
    one_hot = np.zeros((environment.n_states, environment.n_actions))
    one_hot[range(environment.n_states), policy] = 1
    return one_hot


def get_value_function(environment, policy, q_value, greedy=False):
    if greedy:
        policy = greedy_policy(policy)
    # if policy is deterministic, convert to 1 hot vector
    if len(policy.shape) == 1:
        policy = convert_to_one_hot_vector(environment, policy)

    # Return the expected value from Q-values
    return np.sum(q_value * policy, axis=1)


def random_state(environment):
    return np.random.randint(environment.n_states)


def random_action(environment):
    return np.random.randint(environment.n_actions)


def random_policy(environment, stochastic=True):
    if not stochastic:
        policy = np.random.randint(low=environment.n_actions, size=environment.n_states)
    else:
        policy = np.random.rand(environment.n_states, environment.n_actions)
        policy = policy / (np.sum(policy, axis=1, keepdims=True) + 1e-10)
    return policy


def optimal_policy(q_value, greedy=False):
    # check which action gives best expected and assign it
    if greedy:
        return np.argmax(q_value, axis=1)
    else:
        policy = q_value
        policy = policy / (np.sum(policy, axis=1, keepdims=True) + 1e-10)
        return policy


# param can be stochastic policy or Q_Value
def greedy_policy(param):
    if len(param.shape) == 1:
        return param
    return np.argmax(param, axis=1)


def generate_episode(environment, state, action, policy):
    total_reward = 0
    for i in range(environment.episode_length):

        # simulate and get real reward and next state
        reward, state = environment.simulate(state, action)

        # choose an action for next state
        action = choose_action(environment, policy, state)

        total_reward = total_reward + environment.gamma ** i * reward

        # convergence check
        if environment.gamma ** i * reward < 1e-3:
            break

    return total_reward
