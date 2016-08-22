import _ica.default as df
import _ica
import numpy as np

def ica(data, iter=df.iter, verbosity=df.verbosity, nthreads=df.nthreads,
        rho=df.rho, fbatch=df.fbatch, theta=df.theta):
    
    X = np.ascontiguousarray(data)
    NC = X.shape[0]
    weights = np.empty((NC, NC), dtype=X.dtype)
    sphere = np.empty((NC, NC), dtype=X.dtype)
    _ica.ica(data, weights, sphere, iter, verbosity, 
                    nthreads, rho, fbatch, theta)
    W = np.dot(weights, sphere)
    sources = np.dot(W, data)
    return sources, W


def normalize(X):
    range = np.max(X,1) - np.min(X,1)
    return (X - np.mean(X,1,keepdims=True))/range[:,np.newaxis]

def match(X, Y):    
    U = normalize(X)
    V = normalize(Y)
    rmse = lambda X: np.sqrt(np.mean(X**2, axis=1))
    minargmin = lambda x: [np.min(x), np.argmin(x)]
    bestmatch = lambda v: minargmin(np.minimum(rmse(U - v), rmse(U + v)))
    idx = np.array([bestmatch(v) for v in V])
    return U[idx[:,1].astype(int),:], np.mean(idx[:,0])
        
def amari_error(W, A):
    k = W.shape[0]
    P = np.abs(np.dot(W, A))
    rmax = np.max(P, axis=0, keepdims=True)
    cmax = np.max(P, axis=1, keepdims=True)
    E1 = P/rmax
    E2 = P/cmax
    c = 1./(2*k*(k-1))
    return c*(np.sum(E1) + np.sum(E2) - 2*k)

