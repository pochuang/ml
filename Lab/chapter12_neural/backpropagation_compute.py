import numpy as np
import math
import matplotlib.pyplot as plt 
import sys

def sigmoid(h):
    return 1.0 / (1.0 + np.exp(-h))

def cost(t,o):
    return pow((t-o),2)*0.5    

Target = 0.0
Input = 1.0
Weight = 0.3
Bias = 0.5
LearningRate = 0.5
Z = Input * Weight + Bias
Output = sigmoid(Z)
Cost = cost(Target,Output)
X = []
y = []

repeat_times = int(sys.argv[1])
#repeat_times=10000
print repeat_times
for i in range(repeat_times):
    dCdO = Output-Target
    dOdZ = Output*(1-Output)
    dCdZ = dCdO * dOdZ
    dCdB = dCdZ
    dZdW = 1
    dCdW = dCdZ*dZdW
    Weight = Weight - dCdW * LearningRate
    Bias = Bias - dCdB * LearningRate
    Z = Input * Weight + Bias
    Output = sigmoid(Z)
    Cost = cost(Target,Output)

print ('Bias : %10.10f' %Bias) 
print ('Weight : %10.10f' %Weight)
print ('Cost : %10.10f' %Cost)
print ('Output : %10.10f' %Output)