import numpy as np

from Battery import Battery
from ValueIteration import ValueIteration

problem = Battery(reward=[0, 0, 0, 0, 0, 0])

solver = ValueIteration(problem.get_reward_function(), problem.transition_model, gamma=0.9)
solver.train()