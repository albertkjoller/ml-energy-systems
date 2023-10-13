from sklearn.base import BaseEstimator
import numpy as np
from tqdm import tqdm

import pandas as pd

class GradientBasedLinearRegression:

    def __init__(self, regularization = '', lambda_=None):
        # Initialize parameters
        self.bias = 0
        self.loss = []
        self.weight_history = []
        self.regularization = regularization
        self.lambda_        = lambda_           # regularization parameter
        
    def fit(self, X, y, alpha = 0.0001, n_iterations = 1000, tolerence_convergence = 1e-4):
        # Remove intercept
        columns_order   = np.sort(X.columns)
        X               = X[columns_order]

        # Get num observations and num features
        self.n, self.m = X.shape
        # Create array of weights, one for each feature
        self.weights              = np.zeros(self.m)
        self.coefficient_names    = np.concatenate([['Intercept'], columns_order]) 

        # Iterate a number of times
        for _ in tqdm(range(n_iterations), desc="Training Linear Regression..."):
            
            # Generate prediction
            y_hat = np.dot(X, self.weights) + self.bias 

            # Calculate error and loss
            error =  y_hat - y
            
            # Log the loss
            self.loss.append(np.square(error).mean())

            if self.regularization.lower() == 'ridge':
                raise NotImplementedError("Not implemented yet!")
            elif self.regularization.lower() == 'lasso':
                gradient_wrt_weights    = np.dot(X.T, error)
                gradient_wrt_bias       = (1/ self.n) * np.sum(error) 

                for k in range (self.m):
                    if (self.weights[k] > 0):
                        gradient_wrt_weights[k] = (gradient_wrt_weights[k] + self.lambda_/2)/self.n
                    else:
                        gradient_wrt_weights[k] = (gradient_wrt_weights[k] - self.lambda_/2)/self.n
 
            else:
                # Calculate gradients using partial derivatives
                gradient_wrt_weights    = (1 / self.n) * np.dot(X.T, error)
                gradient_wrt_bias       = (1 / self.n) * np.sum(error)    

   
            # Update parameters using gradients and alpha    
            self.weights            = self.weights - alpha * gradient_wrt_weights
            self.bias               = self.bias - alpha * gradient_wrt_bias
            self.coefficients       = np.concatenate([np.array([self.bias]), self.weights])

            # Store weights for convergence analysis
            self.weight_history.append(self.coefficients)
            if np.linalg.norm(self.weights) < tolerence_convergence:
                break
        
    def predict(self, X):
        columns_order   = np.setdiff1d(X.columns, ['Intercept'])
        X               = X[columns_order]        
        # Generate predictions using current weights and bias 
        return np.dot(X, self.weights) + self.bias 

class ClosedFormLinearRegression(BaseEstimator):
    def __init__(self, regularization = '', lambda_=None):
        self.coefficients   = None
        self.regularization = regularization
        self.lambda_        = lambda_           # regularization parameter
        
    def _add_intercept(self, X):
        # Insert intercept as first column in dataframe
        return pd.DataFrame(np.ones((X.shape[0], 1)), columns=['Intercept']).join(X)
        
    def fit(self, X, y):
        X                       = self._add_intercept(X)
        self.coefficient_names  = np.sort(X.columns)
        X                       = X[self.coefficient_names].to_numpy()
        y                       = y.to_numpy()

        if self.regularization.lower() == 'ridge':
            self.coefficients = np.linalg.solve(X.T.dot(X) + self.lambda_ * np.eye(X.shape[1]), X.T.dot(y))
        elif self.regularization.lower() == 'lasso':
            raise NotImplementedError("There is no Closed Form solution for Lasso regularization!")
        else:
            self.coefficients = np.linalg.solve(X.T.dot(X), X.T.dot(y))
        
    def predict(self, X):
        X = self._add_intercept(X)
        X = X[self.coefficient_names].to_numpy()
        return X.dot(self.coefficients)
    

class LocallyWeightedLinearRegression(BaseEstimator):
    def __init__(self, sigma, regularization = '', kernel: str = 'Gaussian', lambda_=0.0):
        self.coefficients   = None
        self.sigma          = sigma
        self.regularization = regularization
        self.lambda_        = lambda_           # regularization parameter
        self.kernel_        = self.get_kernel(kernel)
    
    def get_kernel(self, kernel):
        if kernel == 'Gaussian':
            return self.gaussian_kernel
        else:
            raise NotImplementedError(f'Kernel {self.kernel_} not implemented')
        
    def gaussian_kernel(self, dists, kappa: float = 1.):
        """
        Compute the Gaussian kernel
        """
        return kappa * np.exp(-dists**2 / (2 * self.sigma**2)) 

    def _add_intercept(self, X):
        # Insert intercept as first column in dataframe
        return pd.DataFrame(np.ones((X.shape[0], 1)), columns=['Intercept']).join(X)
        
    def fit(self, X, y):
        X                       = self._add_intercept(X)
        self.coefficient_names  = np.sort(X.columns)
        X                       = X[self.coefficient_names]
        
        # Fit non-parametric model
        self.Xtrain_ = X.to_numpy()
        self.ytrain_ = y.to_numpy()

    def predict(self, X):
        # Add intercept
        X = self._add_intercept(X)
        X = X[self.coefficient_names].to_numpy()

        # Compute kernels
        dists   = np.linalg.norm((X[:, np.newaxis, :] - self.Xtrain_), axis=-1)
        Ks      = self.kernel_(dists)
        
        # Initialize weights
        thetas  = np.zeros(X.shape)
        
        for i, k in enumerate(Ks):
            # Solve for theta locally
            thetas[i, :]    = np.linalg.solve(self.Xtrain_.T.dot(np.diag(k)).dot(self.Xtrain_) + self.lambda_ * np.eye(X.shape[1]), self.Xtrain_.T.dot(np.diag(k)).dot(self.ytrain_))

        # Predict for each point
        return (X * thetas).sum(axis=1)