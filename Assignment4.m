%%
%implementing slice sampler
%try to build in a sampler to implement the log scale
clc;
clear;


y = [1,0,1,1,0,1,1,1,0,1,1,1,1]; %coin flips 
N=length(y);
z=sum(y);
theta=(0:0.001:1);

Bernoulli_LogLikelihood_PDF = @(theta,N,z) z*log(theta)+ (N-z)*log(1-theta);
Log_Prior=@(theta) log(theta);
Posterior=@(theta) (Log_Prior(theta)+Bernoulli_LogLikelihood_PDF(theta,N,z));

Initial=0.5;
Nsamples=1000;
jump=0.01;

[theta_samples] = slicesample (Initial,Nsamples,'logpdf',Posterior,'width',jump);

subplot(1,2,1)
hist(theta_samples,100)
title('Histogram of posterior samples')
xlim([0,1])
xlabel("theta")    
ylabel("theta|y")

subplot(1,2,2)
plot(theta_samples)
xlabel('sample index')
ylabel('theta(i)')
title('Trace plot of posterior samples')

%Confidence interval
mbe_hdi(theta_samples,0.95);

%probability that theta>0.5

SampleSize= length(theta_samples);
p= length(find(theta_samples>=0.5));
PThetay= p/SampleSize;

%Task 2.b

y1=[1,0,0,0,0,0,0,1,1,0]; %gives error ,step out procedure failed

N1=length(y1);
z1=sum(y1);
theta=(0:0.001:1);

Bernoulli_LogLikelihood_PDF1 = @(theta,N1,z1) z1*log(theta)+ (N1-z1)*log(1-theta);
Log_Prior1=@(theta) log(theta);
Posterior1=@(theta) (Log_Prior1(theta)+Bernoulli_LogLikelihood_PDF1(theta,N1,z1));

Initial1=0.5;
Nsamples1=1000;
jump1=0.01;

[theta_samples1] = slicesample (Initial1,Nsamples1,'logpdf',Posterior1,'width',jump1);

% 2.b.1
SampleSize1= length(theta_samples1);
p1= length(find(theta_samples1>=0.5));
PThetay1= p1/SampleSize1;
% result - P(thetay=0.5)= 0.9780, P(thetay=0.5)=0.2190

%comparing two sequences

hist(theta_samples,100,'FaceColor','y');
hold on;

hist(theta_samples1,100);


%2.b.2
for j=1:length(theta_samples)
    dTheta(j)= theta_samples(j)-theta_samples1(j);
end
%Confidence interval
mbe_hdi(dTheta,0.95)
%result 0.0327    0.6992

hist(dTheta,100); 
hold on; 
line(HDI,[0,0],'Color','r','LineWidth',5);
legend('dTheta','HDI');


%%
%recreating fig 6.4
clc;
clear;
z=17;
N=20;

Initial=0.5;
Nsamples=10000;
jump=0.1;

theta=(0:0.001:1);
LF= @(theta,N,z) z*log(theta)+ (N-z)*log(1-theta);

for count=1:length(theta)
likelihoodf(count)=theta(count)^z*(1-theta(count))^(N-z);
end

%column 1
a1=100;
b1=100;
for count=1:length(theta)
priorbeta1(count)=betapdf(theta(count),a1,b1);
end
PF1=@(theta,a1,b1) log(betapdf(theta,a1,b1));
F1=@(theta) (LF(theta,N,z)+PF1(theta,a1,b1));
L1= slicesample (Initial,Nsamples,'logpdf',F1,'width',jump);


subplot(3,3,1)
area(theta,priorbeta1);
title('prior beta;mode=0.5');
ylabel('beta(100,100)');

subplot(3,3,4)
area(theta,likelihoodf);
title('likelihood(bernoulli)');

subplot(3,3,7)
hist(L1,100);
xlim([0,1]);
title('posterior(beta)');
ylabel('beta(117,103)');



%column 2
a2=18.25;
b2=6.75;
for count=1:length(theta)
priorbeta2(count)=betapdf(theta(count),a2,b2);
end
PF2=@(theta,a2,b2) log(betapdf(theta,a2,b2));
F1=@(theta) (LF(theta,N,z)+PF2(theta,a2,b2));

L2= slicesample (Initial,Nsamples,'logpdf',F1,'width',jump);


subplot(3,3,2)
area(theta,priorbeta2);
title('prior beta;mode=0.75');
ylabel('beta(18.25,6.75)');

subplot(3,3,5)
area(theta,likelihoodf);
title('likelihood(bernoulli)');

subplot(3,3,8)
hist(L2,100);
xlim([0,1]);
title('posterior(beta)');
ylabel('beta(35.25,9.75)');

%column 3
a3=1;
b3=1;
for count=1:length(theta)
priorbeta3(count)=betapdf(theta(count),a3,b3);
end
PF3=@(theta,a3,b3) log(betapdf(theta,a3,b3));
F3=@(theta) (LF(theta,N,z)+PF3(theta,a3,b3));

L3= slicesample (Initial,Nsamples,'logpdf',F1,'width',jump);

subplot(3,3,3)
area(theta,priorbeta3);
ylim([0 5]);
title('prior beta');
ylabel('beta(1,1)');

subplot(3,3,6)
area(theta,likelihoodf);
title('likelihood(bernoulli)');

subplot(3,3,9)
hist(L3,100);
xlim([0,1]);
title('posterior(beta)');
ylabel('beta(18,4)');

%
