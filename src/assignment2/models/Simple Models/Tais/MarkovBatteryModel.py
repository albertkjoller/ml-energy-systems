#%% importing modules
import numpy as np
import pandas as pd
#%% Importing price-data
prices = pd.read_csv("/Users/tais/Downloads/ml-energy-systems/src/assignment2/data/day_ahead_prices.csv")
#%% Cleaning price-data
prices = prices[prices['PriceArea'] == 'DK1'][['StartTimeUTC','SpotPriceEUR']]
prices['StartTimeUTC'] = pd.to_datetime(prices['StartTimeUTC'])
prices = prices.set_index('StartTimeUTC')['SpotPriceEUR']
#%% Declaring Markov Model Class

class MarkovBatteryModel:

    def __init__(self, state_space=None, action_space=None, battery_init_state=0, reward_function=None):
        self.state_space = state_space
        self.action_space = action_space
        self.action_tree = pd.DataFrame([])
        self.battery_state = battery_init_state

    def buildActionTree(self,action_dict):
        states = []
        for v in list(action_dict.values()):
            if type(v) != list:
                states.append(v)
            else:
                for w in v:
                    states.append(w)
        
        action_tree = pd.DataFrame(index=states,columns=["Action"])

        for a in list(action_dict.keys()):
            action_tree.loc[action_dict[a]] = a
        
        self.action_tree = action_tree
        
#%% Declaring State Space Class
class StateSpace:

    def __init__(self, values=None, union=True,states_dict=None):

        assert values
        var_names = list(values.keys())

        if union:
            state_grid = np.meshgrid(*[values[v] for v in var_names]) #Generating a mesh-grid (set containing all possible pairs)
            state_grid = np.array([state_grid[i].flatten() for i in range(len(state_grid))]).transpose() #Flattening and transposing mesh-grid for easier use
            self.states = pd.DataFrame(state_grid,[f'State {i+1}' for i in range(state_grid.shape[0])],columns=var_names) #Declare state_names
#%% Declaring Action Space CLass
class ActionSpace:
    def __init__(self, action_names, battery_effects):
        self.actions = pd.DataFrame({'Action':action_names,'Effect':battery_effects}).set_index('Action')
#%% Building State Space and Action Space
values = dict.fromkeys(['SOC','Price'])
values['SOC'] = ['LOW','HIGH']
values['Price'] = ['LOW','HIGH']
s_s = StateSpace(values)


a_n = ["C","D","W"]
b_e = [100,-100,0]
a_s = ActionSpace(a_n,b_e)
#%% Declaring Markov Model
m_m = MarkovBatteryModel(a_s,s_s)
# %% Building action tree
a_d = {'C':'State 1',
       'D':'State 2',
       'W':['State 3','State 4']}

m_m.buildActionTree(a_d)
print(m_m.action_tree)
# %%

