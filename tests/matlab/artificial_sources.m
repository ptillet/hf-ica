N=100000;

t = linspace(-10,10,N);
Mix = rand(4,4);
X = [sin(2*t) ; square(4*t) ; sawtooth(5*t) ; randn(1,N)];
Z = Mix*X;
for i=1:4
subplot(2,4,i)
plot(Z(i,:));
end
drawnow;

tic;
[U, W, Sphere] = dshf_ica(X, struct('S0',100,'verbosity',2));
toc;

subplot(2,1,2);
for i=1:4
subplot(2,4,4+i)
plot(U(i,:));
end
drawnow;
