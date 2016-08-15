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
    A = np.dot(weights, sphere)
    sources = np.dot(A, data)
    return sources, A
