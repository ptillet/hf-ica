%Number of sample points
N=1000;

%Generate artificial signals
t = linspace(-10,10,N);
X = [sin(2*t) ; cos(0.5*t) ; sawtooth(5*t) ; randn(1,N)];

%Generate mixtures
Z = rand(4,4)*X;

%Plots mixtures
for i=1:4
subplot(2,4,i)
plot(Z(i,:));
end
drawnow;

%Initial Sample-Size
options.S0 = 100; 
%Ratio of the sample size used for Hessian-vector products
options.RS = 0.1;
%Maximum Number of Iterations
options.maxIter = 200;
%Verbosity Level:
%0 : No information displayed
%1 : Information about the algorithm used
%2 : Information updated as the algorithm proceeds
options.verbosityLevel = 2;

tic;
%[W, Sphere] = neo_ica(X); %Use defaults
[W, Sphere] = neo_ica(X, options);
toc;

%Unmix
IndependentComponents = W*Sphere*X;

%Plots independent components
subplot(2,1,2);
for i=1:4
subplot(2,4,4+i)
plot(IndependentComponents(i,:));
end
drawnow;
