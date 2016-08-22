import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from neo_ica import ica, match, normalize
from sklearn.decomposition import FastICA

def sawtooth(t):
    return t - np.floor(t)

def square(t):
    return np.sign(np.cos(t))

np.random.seed(0)
# Creates time points
N = 1000
t = np.linspace(-10, 10, N)
# Creates sources
X = np.empty((4,N))
X[0,:] = np.sin(t)
X[1,:] = square(t)
X[2,:] = sawtooth(t)
X[3,:] = np.random.rand(N)
NC = X.shape[0]
# Mix
Y = np.dot(np.random.rand(4, 4), X)
# Unmix
S, _ = ica(Y[:,:])
#S = FastICA().fit_transform(Y.T).T
# Plot
X = normalize(X)
S = normalize(S)
S, error = match(S, X)
plt.subplot(3,1,1)
plt.title('True sources')
for i, color in zip(range(NC), ['r', 'g', 'b', 'm']):
    plt.plot(X[i,:], color)
plt.subplot(3,1,2)
plt.title('Observations')
for i, color in zip(range(NC), ['r', 'g', 'b', 'm']):
    plt.plot(Y[i,:], color)
plt.subplot(3,1,3)
plt.title('Recovered')
for i, color in zip(range(NC), ['r', 'g', 'b', 'm']):
    plt.plot(S[i,:], color)
plt.show()
