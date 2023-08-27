#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
def sigmoid(h):
    return 1.0 / (1.0 + np.exp(-h))


h = np.arange(-5, 5, 0.1)
s_h = sigmoid(h)
tanh_h = np.tanh(h)
relu_h = h * (h > 0)
plt.plot(h, s_h)
plt.plot(h, tanh_h)
plt.plot(h, relu_h)
plt.axvline(0.0, ls='dotted', color='k')
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
plt.axhline(y=0.0, ls='dotted', color='k')
#plt.yticks([0.0, 0.5, 1.0])
plt.yticks([-2.0,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])
plt.ylim(-1.5, 5.0)
plt.show()