from tqdm import tqdm

from common_utils import *
from environment import Environment


def q_learning_policy_evaluation(environment, policy, q_value=None, return_q_value=True, alpha=1e-2):
    # generate zero value function
    if q_value is None:
        q_value = np.zeros((environment.n_states, environment.n_actions))

    for _ in tqdm(range(environment.n_episodes)):
        # select a random starting state
        state = random_state(environment)

        # select a random action and get reward
        action = choose_action(environment, policy, state)

        # simulate for one action and get actual reward for taking this action and
        # transition to a new state in following this action
        actual_reward, next_state = environment.simulate(state, action)

        # update Expected Value assuming we took optimal actions after one step
        q_value[state, action] = q_value[state, action] + alpha * (
                actual_reward + environment.gamma * get_value_function(environment, policy, q_value, greedy=True)[next_state] - q_value[
            state, action])

    if return_q_value:
        return q_value
    else:
        return get_value_function(environment, policy, q_value)


def q_learning(environment):
    # generate random policy to start with
    policy = random_policy(environment)

    q_value = None

    for _ in range(environment.max_iterations):
        # for convergence check
        prev_policy = policy

        # evaluate policy
        q_value = q_learning_policy_evaluation(environment, policy, return_q_value=True)

        # improve policy
        policy = optimal_policy(q_value, greedy=False)

        # if converged, stop
        if np.all(greedy_policy(policy) == greedy_policy(prev_policy)):
            break

    return greedy_policy(policy), get_value_function(environment, policy, q_value)


def main():
    # generate sample environment
    environment = Environment(n_states=5, n_actions=2, n_episodes=50000)

    # evaluate the policy
    policy, value = q_learning(environment)
    print('Q-Learning Policy:', policy)
    print('Q-Learning Values:', value)


if __name__ == '__main__':
    main()
