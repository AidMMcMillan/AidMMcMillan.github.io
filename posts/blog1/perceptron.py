import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self):
        self.w = []
        self.history = []
        
    def fit(self, X, y, max_steps = 1000):
        #Create X_ vector = [X,1]
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)

        #Initiate w vector with random wieghts and bias
        self.w = np.random.rand(X_.shape[1])
        
        for num in range(max_steps):
            i = np.random.randint(0, high=(X_.shape[0]-1)) #Choose random i
            
            y_ = 2*y[i]-1
            
            #Update w using perceptron algorithm
            self.w = self.w + (1*((y_*(self.w@X_[i]))<0))*(y_*X_[i])
            
            self.history.append(self.score(X_, y))
            
            if self.score(X_, y) == 1:
                #Break if acurracy reaches 1
                break
            
    def predict(self, X):
        return 1*((X@self.w)>0)
        
    def score(self, X, y):
        return np.mean(1*(self.predict(X)==y))