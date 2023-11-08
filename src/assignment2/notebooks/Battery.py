import numpy as np

class Battery(object):
    """
    class for simulating battery.
    """

    def __init__(self, reward, id='battery', capacity=500,
                 timeStep=6,
                 minimum_charge=0):
        # self.id = id + '_' + str(Battery.idCounter)
        # Battery.idCounter += 1
        self.capacity = capacity
        self.state = [capacity]
        self.timeStep = 24/timeStep
        self.time = []
        self.minimum_charge = minimum_charge
        self.action = [0]
        self.current = [0]

        self.num_rows = 6 # charged 0, charged 100, charged 200, charged 300, charged 400, charged 500
        self.num_cols = 2 # price high, price low
        self.num_states = self.num_rows * self.num_cols # 12 states {(Charged 0, Price high), (Charged 0, Price low), 
                                            # (Charged 100, Price high), (Charged 100, Price low)}
                                             # (Charged 200, Price high), (Charged 200, Price low)}
        self.num_actions = 3 # charging, discharging, idle
        self.reward = reward
        self.transition_model = self.get_transition_model()

    def charge(self, action):
        self.state.append(self.state[-1] + action)
        self.current.append(action)
        self.action.append(action)

    def discharge(self, action):
        self.state.append(self.state[-1] + action)
        self.current.append(action )
        self.action.append(action)


    def get_reward_function(self):
        # reward of each state
        high_price = 500
        low_price = 20
        self.reward = np.array([[-low_price*100, -high_price * 100, -low_price*100, -high_price * 100, -low_price*100, -high_price * 100,
                                 -low_price*100, -high_price * 100, -low_price*100, -high_price * 100, -low_price*100, -high_price * 100],
                                [+low_price*100, +high_price * 100, +low_price*100, +high_price * 100, +low_price*100, +high_price * 100,
                                 +low_price*100, +high_price * 100, +low_price*100, +high_price * 100, +low_price*100, +high_price * 100],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        return self.reward

    def get_transition_model(self):
        # state can be any value from 0 to 5
        # action any value from 0 to 2
        transition_model = np.zeros((self.num_states,self.num_actions, self.num_states))
        # probability matrix 6 X 6
        probability_matrix = np.array([[0.80, 0.20, 0.80, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # 0, low
                                       [0.85, 0.15, 0.85, 0.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], # 0, high
                                       [0.80, 0.20, 0.80, 0.20, 0.80, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], #100, low
                                       [0.85, 0.15, 0.85, 0.15, 0.85, 0.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], #100 high
                                       [0.00, 0.00, 0.80, 0.20, 0.80, 0.20, 0.80, 0.20, 0.00, 0.00, 0.00, 0.00], #200 low
                                       [0.00, 0.00, 0.85, 0.15, 0.85, 0.15, 0.85, 0.15, 0.00, 0.00, 0.00, 0.00], #200 high
                                       [0.00, 0.00, 0.00, 0.00, 0.80, 0.20, 0.80, 0.20, 0.80, 0.20, 0.00, 0.00], #300 low
                                       [0.00, 0.00, 0.00, 0.00, 0.85, 0.15, 0.85, 0.15, 0.85, 0.15, 0.00, 0.00], #300 high
                                       [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.80, 0.20, 0.80, 0.20, 0.80, 0.20], #400 low
                                       [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.85, 0.15, 0.85, 0.15, 0.85, 0.15], #400 high
                                       [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.80, 0.20, 0.80, 0.20], #500 low
                                       [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.85, 0.15, 0.85, 0.15]]) #500 high
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

    def get_state(self):
        return {'time': self.time[-1], 'state': self.state[-1]}

    def get_history(self):
        return {'time': np.array(self.time), 'state': np.array(self.state),
                'action': np.array(self.action),
                'current': np.array(self.current)}