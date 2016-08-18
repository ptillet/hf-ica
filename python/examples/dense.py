import numpy as np
import matplotlib.pyplot as plt
from neo_ica import ica

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

def random_signal(start, stop, npoints, ncoeffs, minp, maxp, nsigs, tile=4):
    sig = np.empty((nsigs, npoints))
    i = 0
    for i in range(0, nsigs - tile, tile):
        sig[i:i+tile, :] = _random_signal(start,stop,npoints,ncoeffs,minp,maxp,tile)
    sig[i:, :] = _random_signal(start,stop,npoints,ncoeffs,minp,maxp,nsigs-i)
    return sig

#sig = random_signal(start=-50, stop=50, npoints=200000, ncoeffs=10, minp=.1, maxp=10, nsigs=128)
#np.save('sig.npy', sig)
sig = np.load('sig.npy')
sig = sig.astype(np.float32)
S, _ = ica(sig, verbosity=2, iter=100)
#plt.plot(sig.T)
#plt.show()
