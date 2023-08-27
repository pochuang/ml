import numpy as np
import math
import matplotlib.pyplot as plt 

def sigmoid(h):
    return 1.0 / (1.0 + np.exp(-h))

def cost(t,o):
    return pow((t-o),2)*0.5    

#LearningRate = [0.3,0.4,0.5,0.6,0.7,0.8]
LearningRate = [0.3,0.5]
for lr in LearningRate:
    Target = 0.0
    Input = 1.0
    Weight = 0.5
    Bias = -1.0
    Z = Input * Weight + Bias
    Output = sigmoid(Z)
    Cost = cost(Target,Output)
    X = []
    y = []
    for i in range(1000):
        dCdO = Output-Target
        dOdZ = Output*(1-Output)
        dCdZ = dCdO * dOdZ
        dCdB = dCdZ
        dZdW = 1
        dCdW = dCdZ*dZdW
        Weight = Weight - dCdW * lr
        Bias = Bias - dCdB * lr
        Z = Input * Weight + Bias
        Output = sigmoid(Z)
        Cost = cost(Target,Output)
        X.append(i)
        y.append(Cost)    
    
    plt.plot(X,y)


plt.show()