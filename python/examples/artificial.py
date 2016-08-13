import numpy as np
#import matplotlib.pyplot as plt
from neo_ica import ica

N = 10000
t = np.linspace(-10, 10, N)
#creates sources
X = np.empty((4,N))
X[0,:] = np.sin(3*t) + np.cos(6*t)
X[1,:] = np.cos(10*t)
X[2,:] = np.sin(5*t)
X[3,:] = np.random.rand(N)
#Mixing
Y = np.dot(np.random.rand(4, 4), X)
W = np.random.rand(4, 4)
S = np.random.rand(4, 4)
S, weights, sphere = ica(Y.astype(np.float32))
