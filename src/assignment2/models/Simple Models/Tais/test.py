#%% importing modules
import numpy as np
import pandas as pd
import data

#%% Declaring State Space
values = dict.fromkeys(['SOC','Price'])
values['SOC'], values['Price'] = [0,100,200,300,400,500], ['LOW','HIGH']
s_s = StateSpace(values)
#%% Declaring Action Space
a_n, b_e = ["C","D","W"], [100,-100,0]
a_s = ActionSpace(a_n,b_e)
#%% Declaring Markov Model
m_m = MarkovBatteryModel(a_s,s_s)
# %% Building action tree
a_d = {'C':'State 1',
       'D':'State 2',
       'W':['State 3','State 4']}

m_m.buildActionTree(a_d)
print(m_m.action_tree)