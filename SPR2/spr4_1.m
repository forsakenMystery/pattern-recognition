m=[0 0; 0 2]';
S(:,:,1)=[0.2 0; 0 0.2];
S(:,:,2)=[0.2 0; 0 0.2];
P=[1/3 2/3];
N=1000;
randn('seed',0);
[X]=generate_gauss_classes(m,S,P,N);
h=0.1;
pdfx_approx=Parzen_gauss_kernel(X,h,-5,5);
% where is example 1.4.13 but in any case the code would be like: also then
% we decrese ybayes from y which we get from generate then we sum the
% equals of list to 1
yBayesian = bayes_classifier(miangin_pdf,covariance_pdf,P,X);
