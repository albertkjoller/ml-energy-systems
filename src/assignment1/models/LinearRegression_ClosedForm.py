from sklearn.base import BaseEstimator
import numpy as np
# closed form

class LinearRegression_ClosedForm(BaseEstimator):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        
    def _add_intercept(self, X):
        return np.c_[np.ones(X.shape[0]), X] 

    def fit(self, X, y):
        if self.fit_intercept:
            X = self._add_intercept(X)

        self.coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        
        if self.fit_intercept:
            self.intercept_ = self.coefficients[0]
            self.coef_ = self.coefficients[1:]
        else:
            self.coef_ = self.coefficients

    def predict(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)
        return X.dot(self.coefficients)