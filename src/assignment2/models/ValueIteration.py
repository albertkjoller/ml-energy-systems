import numpy as np
import pandas as pd

class ValueIteration:
    def __init__(self, env, gamma):
        
        # Initialize environment-related parameters
        self.env                = env
        self.num_states         = env.transition_model.shape[0]
        self.num_actions        = env.transition_model.shape[1]
        self.reward_function    = np.nan_to_num(env.get_reward_function())
        self.transition_model   = env.transition_model
        
        # Initialize algorithm-related parameters
        self.gamma              = gamma
        self.values             = np.zeros(self.num_states)
        self.policy             = None

    def one_iteration(self):
        # Initialize delta stopping criteria
        delta = 0
        for s in range(self.num_states):
            # Get current state values
            temp = self.values[s]

            # Initialize new state values
            v_list = np.zeros(self.num_actions)
            for a in range(self.num_actions):

                # Extract transition probabilities
                p = self.transition_model[s, a]
                # Compute canditates for new state value
                v_list[a] = self.reward_function[a, s] + self.gamma * np.sum(p * self.values)

            # Store value associated with maxium action
            self.values[s] = max(v_list)
            # Update convergence criteria
            delta = max(delta, abs(temp - self.values[s]))

        return delta

    def train(self, tol=1e-3):
        # Initialize by running a single iteration
        iteration = 0
        delta = self.one_iteration()
        # Keep track of delta history
        self.delta_history = [delta]

        # Run until convergence
        while delta > tol:
            iteration += 1
            delta = self.one_iteration()
            # Store convergence history
            self.delta_history.append(delta)
            if delta < tol:
                break
        
    def get_policy(self):
        # Initialize policy of non-possible actions
        pi = -1 * np.ones(self.num_states)
        pi_actions = {}

        # Compute optimal policy
        for s in range(self.num_states):
            # Initialize new state values
            v_list = np.zeros(self.num_actions)

            # Iterate over all possible actions
            for a in range(self.num_actions):
                # Extract transition probabilities
                p = self.transition_model[s, a]
                # Compute canditates for new state value
                v_list[a] = self.reward_function[a, s] + self.gamma * np.sum(p * self.values)

            # Take action that maximizes the value function
            pi[s] = np.argmax(v_list) 
            pi_actions[s] = pd.DataFrame(v_list, index=self.env.action_space.keys(), columns=[self.env.state_space[s]])
        
        pi = {str(self.env.state_space[i]): self.env.idx2action.get(pi_) for i, pi_ in enumerate(pi.astype(int))}
        return pi, pd.concat(pi_actions, axis=1).T