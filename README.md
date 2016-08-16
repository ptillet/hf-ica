Newton Entropy Optimization (NEO) ICA
=====

Large-scale Infomax-ICA using second-order optimization.

**Quadratic convergence**

Faster and better convergence using Truncated Newton (a.k.a Hessian-free) optimization.
This has the same memory cost as usual gradient descent methods.

http://www.sciencedirect.com/science/article/pii/S037704270000426X
http://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf

**Adaptive minibatch size**

Uses the variance of the approximate likelihood's gradient/hessian-vector product
to adjust the mini-batch size across iterations.

http://www.optimization-online.org/DB_FILE/2011/11/3226.pdf

**Fast implementation**

The algorithm was implemented for CPUs using BLAS, OpenMP and SSE intrinsics.
Approximate math is used when possible (https://github.com/herumi/fmath).

If your hardware does not support SSE intrinsics, NEO-ICA falls back to non-vectorized code.

**Multi-Languages**

C++, Python and MATLAB are supported. 

**Lightweight and portable**

Compiles with both GCC 4.8+  and MSVC 2015.

MATLAB bindings have no dependency (links to MATLAB's BLAS/LAPACK).

Python bindings require Numpy (links to BLAS/LAPACK used by Numpy)<sup>1</sup>.

C++ library requires BLAS/LAPACK<sup>1</sup>.

<sup>1</sup> ICA typically involves very tall and skinny matrices. MKL is AFAIK better than the competition at handling those.
