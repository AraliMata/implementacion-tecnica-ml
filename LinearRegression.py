#Yoceline Aralí Mata Ledezma A01562116

import numpy as np
import pandas as pd
import math

class LinearRegression:

    def __b0(self, X, y):
        summ = 0
        n = len(X)

        for i in range(n):
            term = y[i] - np.dot(X[i].T, self.betas)
            summ += term
        
        b0 = summ / n

        return b0
   
    def fit(self, features, y):
        y = y.to_numpy()
        X = features.to_numpy()
        
        inv =  np.linalg.inv(np.dot(X.transpose(),X))
        self.betas = np.dot(inv, np.dot(X.T,y))
        self.b0 = self.__b0(X, y)

    
    def predict(self, data):
        X = data.to_numpy()
        results = []
        
        for i in range(len(X)):
            result = self.b0
            for j in range(len(self.betas)):
                result += self.betas[j] * X[i][j]
            
            results.append(result)

        self.y_estimated = np.array(results)
        return self.y_estimated
    
    def mse(self, y_actual, y_estimated):
        return np.square(np.subtract(y_actual,y_estimated)).mean()
                  
    def rmse(self, y_actual, y_estimated):
        return math.sqrt(self.mse(y_actual,y_estimated))
