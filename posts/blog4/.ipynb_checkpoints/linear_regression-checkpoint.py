import numpy as np
from matplotlib import pyplot as plt

def pad(X):
    return np.append(X, np.ones((X.shape[0], 1)), 1)

def make_data(n_train = 100, n_val = 100, p_features = 1, noise = .1, w = None):
    if w is None: 
        w = np.random.rand(p_features + 1) + .2
    
    X_train = np.random.rand(n_train, p_features)
    y_train = pad(X_train)@w + noise*np.random.randn(n_train)

    X_val = np.random.rand(n_val, p_features)
    y_val = pad(X_val)@w + noise*np.random.randn(n_val)
    
    return X_train, y_train, X_val, y_val

def draw_line(w, x_min, x_max, clr, axarr):
        x = np.linspace(x_min, x_max, 101)[:,np.newaxis]
        X = pad(x)
        for plot in axarr:
            plot.plot(x, X@w, color = clr)

class LinearRegression:
    def __init__(self):
        self.w = []
        self.score_history = []

    def predict(self, X):
        return X@self.w
    
    def score(self, X, y):
        y_hat = self.predict(X)
        y_bar = np.mean(y)
        num = np.sum((y_hat-y)**2)
        den = np.sum((y_bar-y)**2)
        
        return 1-num/den
    
    def gradient(self, P, q):
        return 2*(P@self.w-q)
        
    def fit(self, X, y, method = "analytic", alpha = 0.001, max_epochs = 1000):
        X_ = pad(X)
        P = X_.T@X_
        q = X_.T@y
        if method == "analytic":
            self.w = np.linalg.inv(P)@q
            
        if method == "gradient":
            self.w = np.random.rand(X_.shape[1])
            
            for num in range(max_epochs):
                self.score_history.append(self.score(X_, y))
                
                self.w -= self.gradient(P, q) * alpha