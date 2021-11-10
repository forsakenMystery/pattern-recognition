m = [-5,5;5,-5; 5,5; -5,-5]';
s=3;
N=100;
rng(0);
[x1 y1] = data_generator(m,s,N);
rng(10);
[x2 y2] = data_generator(m,s,N);
s = 5;
rng(0);
[x3 y3] = data_generator(m,s,N);
rng(10);
[x4 y4] = data_generator(m,s,N);

input = x1;
target = y1;
test = x2;
testtarget = y2;
 
% input = x3;
% target = x3;
% test = x4;
% testtarget = y4;

[ss si accuracy_train accuracy_test] = SVM_clas(input, target, test, testtarget, .001, 1, .5)