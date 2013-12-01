Dynamically Sampled Hessian Free (DSHF) ICA
=====

Large-scale Infomax-ICA using second-order information.

##Features

* Adaptive to the redundancy in the signals

Proceeds in a mini-batch fashion, and dynamically adjust the sample-size
as the algorithm proceeds.

* Fast convergence

Numerous algorithms are implemented (Hessian-Free, L-BFGS, NCG), but in
our experiments Hessian-Free optimization almost always performs best

* Fast implementation

The algorithm was implemented using OpenBlas, OpenMP and Intel's SSE3.
It's fast. GPUs were not retained because of the number of source
signals being usually too small to justify their use.

* MATLAB Wrapper available

A MATLAB MEX wrapper is available. It still requires OpenBlas as of now
(ie we don't use mwBlas), but I will try to offer precompiled binaries
ASAP.
