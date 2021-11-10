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

hidden = 2;
num_epochs = 300;
train_fun = 'traingda';
mu = .001;
max_xy = max(x1,[],2);
min_xy = min(x1,[],2);

net1 = NN_training(input, target, hidden, mu, num_epochs, train_fun);
net1 = train(net1, input, target);
net1 = train(net1, input, target);

h = plot_dec_region(net1,[-5,5;5,-5; 5,5; -5,-5]',[min_xy(1),max_xy(1),min_xy(2),max_xy(2)],.01);
x1_target = round(net1(input));
x2_target = round(net1(test));
x1_class_results = x1_target(1,:);
x1_class = testtarget(1,:);
x2_class_results = x2_target(1,:);
x2_class = testtarget(1,:);
x1_target_error_2 = confused(x1_class, x1_class_results);
x2_target_error_2 = confused(x2_class, x2_class_results);

nodes2x1 = sprintf('Adaptive Learning Rate Error train:  %0.2f%%', x1_target_error_2*100);
nodes2x2 = sprintf('Adaptive Learning Rate Error test:  %0.2f%%', x2_target_error_2*100);
disp(nodes2x1);disp(nodes2x2);