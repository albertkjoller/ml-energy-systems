import numpy as np
from collections import defaultdict

class Battery(object):
    """
    class for simulating battery.
    """

    def __init__(self, prices, high_low_prices, id='battery'):
        # Define state and action spaces
        self.state_space        = [str(f"[{c*100}, {p}]") for c in range(6) for p in ['L', 'H']]
        self.num_states         = len(self.state_space)
        self.boundaries         = {'soc': {'min': 0, 'max': 500}, 'price': {'min': 'L', 'max': 'H'}}

        self.idx2action         = {0: 'Charge', 1: 'Discharge', 2: 'Idle'}
        self.action_space       = {'Charge': 100, 'Discharge': -100, 'Idle': 0}
        self.num_actions        = len(self.action_space) 
        
        # Define price signal
        self.prices             = prices
        self.high_price         = high_low_prices.loc['H']
        self.low_price          = high_low_prices.loc['L']
        
        # Reset battery environment
        self.state              = self.reset()
        # Define transition probabilities
        self.transition_model   = self.get_transition_model()

    def reset(self, init_soc=500):
        self.memory             = defaultdict(dict)
        self.time               = 0
        self.state              = [init_soc, self.prices['upcoming_price_level'][self.time + 1]]

        return self.state
        
    def charge(self):
        self.state[0] = np.clip(self.state[0] + 100, self.boundaries['soc']['min'], self.boundaries['soc']['max'])

    def discharge(self):
        self.state[0] = np.clip(self.state[0] - 100, self.boundaries['soc']['min'], self.boundaries['soc']['max'])

    def idle(self):
        pass 
    
    def step(self, action):
        # Get reward
        soc_update  = np.clip(self.state[0] + self.action_space[action], self.boundaries['soc']['min'], self.boundaries['soc']['max'])
        reward      = - self.action_space[action] * self.prices['SpotPriceDKK'][self.time] if (soc_update != 0 or self.state != 100) else 0

        # Update state
        if action == 'Charge':
            self.charge()
        elif action == 'Discharge':
            self.discharge()
        else:
            self.idle()

        # Update time and
        self.time += 1
        # Update price level
        self.state[1] = self.prices['upcoming_price_level'][self.time + 1]

        # Return reward
        return self.state, reward

    def get_reward_function(self):
        # charging, discharging, waiting
        self.reward = np.array([[
            -self.low_price*100, -self.high_price * 100, -self.low_price*100, -self.high_price * 100, -self.low_price*100, -self.high_price * 100, 
            -self.low_price*100, -self.high_price * 100, -self.low_price*100, -self.high_price * 100, -np.inf, -np.inf
        ], [
            -np.inf, -np.inf, self.low_price*100, self.high_price * 100, self.low_price*100, self.high_price * 100,
            self.low_price*100, self.high_price * 100, self.low_price*100, self.high_price * 100, self.low_price*100, self.high_price * 100
        ], [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]])
        return self.reward

    def get_transition_model(self):
        # state can be any value from 0 to 5
        # action any value from 0 to 2
        transition_model = np.zeros((self.num_states,self.num_actions, self.num_states))
        # probability matrix 6 X 6
        probability_matrix = np.array([[0.87, 0.13, 0.87, 0.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # 0, low
                                       [0.15, 0.85, 0.15, 0.85, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # 0, high
                                       [0.87, 0.13, 0.87, 0.13, 0.87, 0.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], #100, low
                                       [0.15, 0.85, 0.15, 0.85, 0.15, 0.85, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], #100 high
                                       [0.00, 0.00, 0.87, 0.13, 0.87, 0.13, 0.87, 0.13, 0.00, 0.00, 0.00, 0.00], #200 low
                                       [0.00, 0.00, 0.15, 0.85, 0.15, 0.85, 0.15, 0.85, 0.00, 0.00, 0.00, 0.00], #200 high
                                       [0.00, 0.00, 0.00, 0.00, 0.87, 0.13, 0.87, 0.13, 0.87, 0.13, 0.00, 0.00], #300 low
                                       [0.00, 0.00, 0.00, 0.00, 0.15, 0.85, 0.15, 0.85, 0.15, 0.85, 0.00, 0.00], #300 high
                                       [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.87, 0.13, 0.87, 0.13, 0.87, 0.13], #400 low
                                       [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.15, 0.85, 0.15, 0.85, 0.15, 0.85], #400 high
                                       [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.87, 0.13, 0.87, 0.13], #500 low
                                       [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.15, 0.85, 0.15, 0.85], #500 high
                                       ])
        
        for s in range(self.num_states):
            # {0 - (Charged 0, Price high), 1 - (Charged 0, Price low), 
            #  2 - (Charged 100, Price high), 3 - (Charged 100, Price low),
            #  4 - (Charged 200, Price high), 5 - (Charged 200, Price low)}
            for a in range(self.num_actions):
                # for each state define the probability of reaching a new state applying action a
                # when charging and state is not maxim
                if a == 0 and s < self.num_states - 2:
                    # state -> state + 2, state + 3/state +1
                    # (charged, price low) -> {(charged +100, price low), (charged +100, price high)}
                    if (s % 2 == 0):
                        transition_model[s, a, s+2] = probability_matrix[s, s+2]
                        transition_model[s, a, s+3] = probability_matrix[s, s+3]
                    else:
                        # (charged, price high) -> {(charged +100, price low), (charged +100, price high)}
                        transition_model[s, a, s+1] = probability_matrix[s, s+1]
                        transition_model[s, a, s+2] = probability_matrix[s, s+2]
                # when discharging    
                elif a == 1 and s > 1:
                    # state -> state -2, state-1/state-2
                    # (charged, price low) -> {(charged - 100, price low), (charged - 100, price high)}
                    if (s % 2 == 0):
                        transition_model[s, a, s-1] = probability_matrix[s, s-1]
                        transition_model[s, a, s-2] = probability_matrix[s, s-2]
                    else:
                        # (charged, price high) -> {(charged +100, price low), (charged +100, price high)}
                        transition_model[s, a, s-2] = probability_matrix[s, s-2]
                        transition_model[s, a, s-3] = probability_matrix[s, s-3]
                #when idle
                elif a == 2:
                    #state -> state, state+1/state-1
                    # (charged, price low) -> {(charged, price low), (charged, price high)}
                    if (s % 2 == 0):
                        transition_model[s, a, s] = probability_matrix[s, s]
                        transition_model[s, a, s+1] = probability_matrix[s, s+1]
                    else:
                        # (charged, price high) -> {(charged +100, price low), (charged +100, price high)}
                        transition_model[s, a, s] = probability_matrix[s, s]
                        transition_model[s, a, s-1] = probability_matrix[s, s-1]
        return transition_model