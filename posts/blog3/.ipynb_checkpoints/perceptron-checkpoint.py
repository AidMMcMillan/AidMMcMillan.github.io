import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self):
        self.w = []
        self.history = []
        
    def fit(self, X, y, max_steps = 1000):
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        
        self.w = np.random.rand(X_.shape[1])
        
        for num in range(max_steps):
            i = np.random.randint(0, high=(X_.shape[0]-1))
            
            y_ = 2*y[i]-1
            
            self.w = self.w + (1*((y_*(self.w@X_[i]))<0))*(y_*X_[i])
            
            self.history.append(score(X_, y))
            
            if score(X_, y) == 1:
                break
            
    def predict(self, X):
        return 1*((X@self.w)>0)
        
    def score(X, y):
        return np.mean(1*((self.predict(X)*y)>0))