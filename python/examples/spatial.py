import scipy.misc as msc
import numpy as np
from neo_ica import ica
import matplotlib.pyplot as plt

def normalize(X, a, b, axis=1):
    X = (X - np.min(X,axis,keepdims=True))/(np.max(X,axis,keepdims=True)- np.min(X,axis,keepdims=True))
    X = (a + X*(b-a)).astype(int)
    return X
    
np.random.seed(0)
plt.gray()
# Creates sources
ascent = msc.ascent()
face = msc.face(gray=True)[256:768,256:768]
noise = np.random.poisson(100,face.shape).astype(float)
# Mix sources
X = np.vstack((x.ravel() for x in [ascent, face, noise]))
Y = np.dot(np.random.rand(3,3), X)
# Recover sources
S, W = ica(Y)
# Normalize between 0 and 255
S = normalize(S, 0, 255)
# Plot
plt.subplot(3,3,2)
plt.title('True sources')
for i in range(3):
    plt.subplot(3,3,i+1)
    plt.imshow(X[i,:].reshape(512,512))
plt.subplot(3,3,5)
plt.title('Observations')
for i in range(3):
    plt.subplot(3,3,4+i)
    plt.imshow(Y[i,:].reshape(512,512))
plt.subplot(3,3,8)
plt.title('Retrieved sources')
#Reorder for prettier plot
S = S[[0,2,1],:]
for i in range(3):
    plt.subplot(3,3,7+i)
    plt.imshow(S[i,:].reshape(512,512))
plt.tight_layout()
plt.show()
