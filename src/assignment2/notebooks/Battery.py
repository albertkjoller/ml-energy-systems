import numpy as np

class Battery(object):
    """
    class for simulating battery.
    """

    def __init__(self, reward, random_rate, id='battery', capacity=500,
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

        self.num_rows = 2 # charged high, charged low
        self.num_cols = 2 # price high, price low
        self.num_states = self.num_rows * self.num_cols # 4 states {(Charged high, Price high), (Charged high, Price low), 
                                            # (Charged low, Price high), (Charged low, Price low)}
        self.num_actions = 3 # charging, discharging, idle
        self.random_rate = random_rate
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
        return self.reward

    def get_transition_model(self):
        
        transition_model = np.zeros((self.num_states, self.num_actions, self.num_states))
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                # {0 - (Charged high, Price high), 1 - (Charged high, Price low), 
                #  2 - (Charged low, Price high), 3 - (Charged low, Price low)}
                s = self.get_state_from_pos((r, c))
                for a in range(self.num_actions):
                    # for each state define the probability of reaching a new state applying action a
                    # when charging
                    if a == 0:
                        # (charged low, price high) -> {(charged high, price high), (charged high, price low)}
                        # (charged low, price low) ->  {(charged high, price high), (charged high, price low)}
                    # when discharging    
                    elif a == 1:
                        # (charged high, price high) -> {(charged low, price high), (charged low, price low)}
                        # (charged high, price low) -> {(charged low, price high), (charged low, price low)}
                    #when idle
                    elif a == 2:
                         # (charged high, price high) -> {(charged high, price high), (charged high, price low)}
                         # (charged high, price low) -> {(charged high, price high), (charged high, price low)}
                         # (charged low, price high) -> {(charged low, price high), (charged low, price low)}
                         # (charged low, price low) -> {(charged low, price high), (charged low, price low)}
                         
                    transition_model[s, a, 0] += 1 - self.random_rate
                    transition_model[s, a, 1] += self.random_rate / 2.0
                    transition_model[s, a, 2] += self.random_rate / 2.0
                    transition_model[s, a, 3] += self.random_rate / 2.0

        return transition_model

    def get_state(self):
        return {'time': self.time[-1], 'state': self.state[-1]}

    def get_history(self):
        return {'time': np.array(self.time), 'state': np.array(self.state),
                'action': np.array(self.action),
                'current': np.array(self.current)}