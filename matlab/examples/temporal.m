%Number of sample points
N=100000;
rng(0,'twister')

%Generate artificial signals
t = linspace(-10,10,N);
X = [sin(2*t) ; cos(0.5*t) ; cos(5*t) ; randn(1,N)];

%Generate mixtures
Z = rand(4,4)*X;

%Plots mixtures
for i=1:4
subplot(2,4,i)
plot(Z(i,:));
end
drawnow;

X = double(X);
tic;
[W, Sphere] = runica(X, 'bias', 'off', 'extended', 0);
[W, fval, k] = rtr_ica(X, struct('nl_func',2, 'nl_rate', 1, 'max_loop', 200, 'whitened', 0, 'etaF', 1e-12, 'etaW', 1e-4));
[W, Sphere] = neo_ica(X, struct('verbose',1,'extended',0));
toc;

IndependentComponents = W*X;
%Plots independent components
subplot(2,1,2);
for i=1:4
subplot(2,4,4+i)
plot(IndependentComponents(i,:));
end
drawnow;
