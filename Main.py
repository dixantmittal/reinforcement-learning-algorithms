import q_learning as q
import value_iteration as vi

# generate sample environment
from common_utils import *
from environment import Environment

environment = Environment(n_states=5, n_actions=2, n_episodes=10000)

print('Q-learning: ')
policy, value = q.q_learning(environment)

print(value)
print(policy)

print('VI: ')
policy, value = vi.value_iteration(environment)

print(value)
print(policy)
