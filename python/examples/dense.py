import numpy as np
import matplotlib.pyplot as plt
from neo_ica import ica, amari_error, match, normalize
from sklearn.decomposition import FastICA
from time import time

np.random.seed(0)

def _random_signal(start, stop, npoints, ncoeffs, minp, maxp, nsigs):
    t = np.expand_dims(np.linspace(start, stop, npoints), 0)
    n = np.expand_dims(np.arange(1, ncoeffs+1), 1)
    Pc = minp + np.random.rand(nsigs)*(maxp - minp)
    phic = np.random.rand(nsigs)*Pc
    Ps = minp + np.random.rand(nsigs)*(maxp - minp)
    phis = np.random.rand(nsigs)*Ps
    a0 = np.random.rand(nsigs)
    an = np.expand_dims(np.random.rand(ncoeffs, nsigs), 1)
    bn = np.expand_dims(np.random.rand(ncoeffs, nsigs), 1)
    nt = np.expand_dims(n*t,2)
    sig = a0 + np.sum(an*np.cos(nt/Pc + phic),0) + np.sum(bn*np.sin(nt/Ps + phis),0)
    return sig.T

def random_signal(start, stop, npoints, ncoeffs, minp, maxp, nsigs, tile=32):
    sig = np.empty((nsigs, npoints))
    i = 0
    for i in range(0, nsigs - tile, tile):
        sig[i:i+tile, :] = _random_signal(start,stop,npoints,ncoeffs,minp,maxp,tile)
    sig[i:, :] = _random_signal(start,stop,npoints,ncoeffs,minp,maxp,nsigs-i)
    return sig

def skewness(X, axis=1):
    mu, std = np.mean(X, axis=axis, keepdims=True), np.std(X, axis=axis, keepdims=True)
    return np.mean(((X - mu)/std)**3, axis=axis)

def sort_skewness(X, axis=1):
    return np.sort(X, axis=axis, order=skewness(X, axis=axis))

ntrials = 20
tneo, errneo = [], []
tfast, errfast = [], []
fastica = FastICA()
for i in range(ntrials):
    #Sources
    X = random_signal(start=0, stop=20, npoints=100000, ncoeffs=10, minp=0, maxp=1, nsigs=32)
    #np.save('X.npy', X)
    #X = np.load('X.npy')
    X = X.astype(np.float32)
    #Observations
    d = X.shape[0]
    A = np.random.rand(d, d)
    Y = np.dot(A, X)
    #Restored - NEO-ICA
    start = time()
    S, W = ica(Y, verbosity=0, extended=False)
    tneo.append(time() - start)
    errneo.append(amari_error(W, A))
    #Restored - FAST-ICA
    start = time()
    S = fastica.fit_transform(Y.T).T
    W = fastica.components_
    tfast.append(time() - start)
    errfast.append(amari_error(W, A))
    #Display
    print 'NEO-ICA: {:.3f}, {:.3f}, {:.3f} / {:.3f} [{:.3f}s]'.format(np.mean(errneo), np.min(errneo), np.max(errneo), np.std(errneo), np.mean(tneo))
    print 'FAST-ICA: {:.3f}, {:.3f}, {:.3f} / {:.3f} [{:.3f}s]'.format(np.mean(errfast), np.min(errfast), np.max(errfast), np.std(errfast), np.mean(tfast))
    print '------------'
#plt.subplot(2,1,1)
#plt.plot(X.T)
#plt.subplot(2,1,2)
#plt.plot(S.T)
#plt.show()
