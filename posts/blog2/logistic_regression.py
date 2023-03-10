import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self):
        self.w = []
        self.loss_history = []
        self.score_history = []
    
    def pad(self, X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)
        
    def sigmoid(self, z):
        z = np.clip(z, -500, 500) #clip added in order to avoid overload errors
        
        return 1 / (1 + np.exp(-z))
    
    def gradient(self, X, y, d_loss):
        #gradient calculation where d_loss is the derivative of the loss function in use
        return np.mean(np.swapaxes(X, 0, 1)*d_loss(X@self.w, y), axis = 1) 

    def predict(self, X):
        return 1*((X@self.w)>0)
    
    def score(self, X, y):
        return np.mean(1*(self.predict(X)==y))
    
    def empirical_risk(self, X, y, loss):
        return loss(X@self.w, y).mean()
    
    def logistic_loss(self, y_hat, y):
        #1e-10 added in case input values are too small
        return -y*np.log(self.sigmoid(y_hat) + 1e-10) - (1-y)*np.log(1-self.sigmoid(y_hat) + 1e-10)
    
    #derivative of the logistic loss function
    def d_logistic_loss(self, y_hat, y):
        return self.sigmoid(y_hat) - y
        
    def fit(self, X, y, alpha = 0.1, max_epochs = 1000):
        #add column of 1s to end of X
        X_ = self.pad(X)

        #initiate w vector with random wieghts and bias
        self.w = np.random.rand(X_.shape[1]) * 2 - 1
        
        prev_loss = 0
        
        for num in range(max_epochs):  
            new_loss = self.empirical_risk(X_, y, self.logistic_loss) #compute loss using empirical risk
            
            self.loss_history.append(new_loss) #add loss to history
            self.score_history.append(self.score(X_, y)) #add score to history
            
            self.w -= alpha * self.gradient(X_, y, self.d_logistic_loss) #gradient step
            
            # check if loss hasn't changed much and terminate if so
            if np.isclose(new_loss, prev_loss):          
                break
            else:
                prev_loss = new_loss
                
    def fit_stochastic(self, X, y, batch_size = 10, alpha = 0.1, max_epochs = 1000, momentum = False):
        #add column of 1s to end of X
        X_ = self.pad(X)

        #initiate w vector with random wieghts and bias
        self.w = np.random.rand(X_.shape[1]) * 2 - 1
        
        n = X.shape[0]
        
        prev_loss = 0
        
        for num in range(max_epochs):
            new_loss = self.empirical_risk(X_, y, self.logistic_loss) #compute loss using empirical risk
            
            self.loss_history.append(new_loss) #add loss to history
            self.score_history.append(self.score(X_, y)) #add score to history
            
            #make vector containing numbers 0-n and then randomly shuffle the numbers
            order = np.arange(n)
            np.random.shuffle(order)
            
            momentum_step = 0
            
            #iterate through batches made of the randomly shuffled order vector
            for batch in np.array_split(order, n // batch_size + 1):
                
                X_batch = X_[batch,:]
                y_batch = y[batch]
                
                prev_w = self.w
                
                #take gradient step calculated using the batch
                self.w = (self.w - alpha * self.gradient(X_batch, y_batch, self.d_logistic_loss) 
                          + (momentum) * momentum_step) # add momentum if activated

                momentum_step = 0.8 * (self.w - prev_w) # momentum step calculation using prev and current w
            
            #check if loss hasn't changed much and terminate if so
            if np.isclose(new_loss, prev_loss):          
                break
            else:
                prev_loss = new_loss
