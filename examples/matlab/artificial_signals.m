numtrainingcases=2000;
t = linspace(0,10,numtrainingcases);
orig_data = [sin(500*t) + 0.2*cos(100*t) ;
                sin(10*t).*cos(30*t)
                cos(10*t)
                sin(t)] ;
orig_data = 0.01*orig_data;
numdim = size(orig_data,1);
A = [0.2 0.3 0.8 0.4;
	0.1 0.5 0.2 0.3;
	0.4 0.7 0.7 0.1;
	0.6 0.1 0.2 0.3];
data = A*orig_data;
hold on;

for i=1:numdim
subplot(4,numdim,i);
hold on;
plot(orig_data(i,:)./max(orig_data(i,:)));
end
drawnow;

hold on;
for i=1:numdim
subplot(4,numdim,numdim*1+i);
hold on;
plot(data(i,:)./max(data(i,:)));
end
drawnow;

tic
independent = linear_parica(data);
toc
plot(independent)
hold on;
for i=1:numdim
subplot(4,numdim,numdim*2+i);
hold on;
plot(independent(i,:)./max(independent(i,:)));
end
drawnow;
