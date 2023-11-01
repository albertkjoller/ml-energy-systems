import numpy as np

from Battery import Battery
from ValueIteration import ValueIteration

problem = Battery(reward=[1/4, 1/8, 1/16, 1/32], random_rate=0.2)

solver = ValueIteration(problem.reward_function, problem.transition_model, gamma=0.9)
solver.train()