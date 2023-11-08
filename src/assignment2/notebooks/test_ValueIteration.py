import numpy as np
import matplotlib.pyplot as plt

from Battery import Battery
from ValueIteration import ValueIteration

problem = Battery(reward=[0, 0, 0, 0, 0, 0])

solver = ValueIteration(problem.get_reward_function(), problem.transition_model, gamma=0.99)
solver.train()