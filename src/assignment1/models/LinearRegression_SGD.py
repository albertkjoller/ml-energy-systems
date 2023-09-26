import numpy as np

class LinearRegression_SGD:
    
    def __init__(self):
        # Initialize parameters
        self.bias = 0
        
        # Create an attribute to log the loss
        self.loss = []
        
    def fit(self, X, y, alpha = 0.0001, n_iterations = 1000, tolerence_convergence = 1e-4):

        # Get num observations and num features
        self.n, self.m = X.shape
        
        # Create array of weights, one for each feature
        self.weights = np.zeros(self.m)
        
        # Iterate a number of times
        for _ in range(n_iterations):
            
            # Generate prediction
            y_hat = np.dot(X, self.weights) + self.bias 
            
            # Calculate error
            error =  y_hat - y
            
            # Calculate loss (mse)
            mse = np.square(error).mean()
            
            # Log the loss
            self.loss.append(mse)

            # Calculate gradients using partial derivatives
            gradient_wrt_weights = (1 / self.n) * np.dot(X.T, error)
            gradient_wrt_bias = (1 / self.n) * np.sum(error)         
                
            # Update parameters using gradients and alpha    
            self.weights = self.weights - alpha * gradient_wrt_weights
            self.bias = self.bias - alpha * gradient_wrt_bias

            if np.linalg.norm(self.weights) < tolerence_convergence:
                break
    
    def predict(self, X):
        # Generate predictions using current weights and bias 
        return np.dot(X, self.weights) + self.bias 