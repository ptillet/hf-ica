import _ica.default as df
import _ica
import numpy as np

def ica(data, iter=df.iter, verbose=df.verbose, nthreads=df.nthreads,
        rho=df.rho, fbatch=df.fbatch, theta=df.theta, extended=df.extended, 
        tol=df.tol):
    
    X = np.ascontiguousarray(data)
    NC = X.shape[0]
    weights = np.empty((NC, NC), dtype=X.dtype)
    sphere = np.empty((NC, NC), dtype=X.dtype)
    _ica.ica(data, weights, sphere, iter, verbose, 
                    nthreads, rho, fbatch, theta, extended, tol)
    W = np.dot(weights, sphere)
    sources = np.dot(W, data)
    return sources, W


def normalize(X):
    range = np.max(X,1) - np.min(X,1)
    return (X - np.mean(X,1,keepdims=True))/range[:,np.newaxis]

def reorder(S, W, A):    
    P = np.abs(np.dot(W, A))
    order = np.argmax(P, axis=0)
    S = S[order,:]
    return S
        
def amari_error(W, A):
    k = W.shape[0]
    P = np.abs(np.dot(W, A))
    rmax = np.max(P, axis=0, keepdims=True)
    cmax = np.max(P, axis=1, keepdims=True)
    E1 = P/rmax
    E2 = P/cmax
    c = 1./(2*k*(k-1))
    return c*(np.sum(E1) + np.sum(E2) - 2*k)

def mean_corr(W, A, S, X):
    S = normalize(S)
    X = normalize(X)
    reorder(S, W, A)
    err = [abs(np.corrcoef(s, x)[0,1]) for s, x in zip(S, X)]
    return np.mean(err)

