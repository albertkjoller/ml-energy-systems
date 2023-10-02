from sklearn.base import BaseEstimator
import numpy as np
from tqdm import tqdm

import pandas as pd

class GradientBasedLinearRegression:

    def __init__(self):
        # Initialize parameters
        self.bias = 0
        self.loss = []
        self.weight_history = []
        
    def fit(self, X, y, alpha = 0.0001, n_iterations = 1000, tolerence_convergence = 1e-4):
        # Remove intercept
        columns_order   = np.sort(X.columns)
        X               = X[columns_order]

        # Get num observations and num features
        self.n, self.m = X.shape
        # Create array of weights, one for each feature
        self.weights            = np.zeros(self.m)
        self.coefficient_names  = np.concatenate([['Intercept'], columns_order]) 

        # Iterate a number of times
        for _ in tqdm(range(n_iterations), desc="Training Linear Regression..."):
            
            # Generate prediction
            y_hat = np.dot(X, self.weights) + self.bias 

            # Calculate error and loss
            error =  y_hat - y
            
            # Log the loss
            self.loss.append(np.square(error).mean())

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
        X                       = X[self.coefficient_names]

        if self.regularization.lower() == 'ridge':
            self.coefficients = np.linalg.solve(X.T.dot(X) + self.lambda_ * np.eye(X.shape[1]), X.T.dot(y))
        elif self.regularization.lower() == 'lasso':
            raise NotImplementedError("Lasso not implemented yet")
        else:
            self.coefficients = np.linalg.solve(X.T.dot(X), X.T.dot(y))
        
    def predict(self, X):
        X = self._add_intercept(X)
        X = X[self.coefficient_names]
        return X.dot(self.coefficients)
    
class LocallyWeightedLinearRegression(BaseEstimator):
    def __init__(self, sigma, regularization = '', lambda_=None):
        self.coefficients   = None
        self.sigma          = sigma
        self.regularization = regularization
        self.lambda_        = lambda_           # regularization parameter

    def gaussian_kernel(self, x, X, sigma, kappa: float = 1.):
        """
        Compute the Gaussian kernel
        """
        return kappa * np.exp(-np.linalg.norm(x - X, ord=2, axis=1)**2 / (2 * sigma**2)) 

    def compute_weight_matrix(self, X):
        # TODO: consider speeding this up using vectorization or something
        N      = len(X)
        self.W = np.zeros((N, N))
        for i in tqdm(range(N), desc="Computing weight matrix"):
            self.W[i, :] = self.gaussian_kernel(X.iloc[i], X, sigma=self.sigma, kappa=1.)

        assert np.all(self.W == self.W.T), "The matrix is not symmetric - something is wrong"

    def fit(self, X, y):
        self.compute_weight_matrix(X)
        if self.regularization.lower() == 'ridge':
            self.coefficients = np.linalg.solve(X.T.dot(self.W).dot(X) + self.lambda_ * np.eye(X.shape[1]), X.T.dot(y))
        elif self.regularization.lower() == 'lasso':
            raise NotImplementedError("Lasso not implemented yet")
        else:
            self.coefficients = np.linalg.solve(X.T.dot(self.W).dot(X), X.T.dot(self.W).dot(y))

    def predict(self, X):
        return X.dot(self.coefficients)
