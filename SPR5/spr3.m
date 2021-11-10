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

hidden = 4;
train_fun = 'traingd';

max_xy = max(x1,[],2);
min_xy = min(x1,[],2);

epochs1 = 300;
lr1 = .01;
net1 = NN_training(input, target, hidden, lr1, epochs1, train_fun);
net1 = train(net1, input, target);
h = plot_dec_region(net1,[-5,5;5,-5; 5,5; -5,-5]',[min_xy(1),max_xy(1),min_xy(2),max_xy(2)],.01);
x1_target = round(net1(input));
x2_target = round(net1(test));
x1_class_results = x1_target(1,:);
x1_class = target(1,:);
x2_class_results = x2_target(1,:);
x2_class = testtarget(1,:);
x1_target_error_2 = confused(x1_class, x1_class_results);
x2_target_error_2 = confused(x2_class, x2_class_results);

lr2 = .001;
epochs2 = 300;
net2 = NN_training(input, target, hidden, lr2, epochs2, train_fun);
net2 = train(net2, input, target);
h = plot_dec_region(net2,[-5,5;5,-5; 5,5; -5,-5]',[min_xy(1),max_xy(1),min_xy(2),max_xy(2)],.01);
x1_target = round(net2(input));
x2_target = round(net2(test));
x1_class_results = x1_target(1,:);
x1_class = target(1,:);
x2_class_results = x2_target(1,:);
x2_class = testtarget(1,:);
x1_target_error_4 = confused(x1_class, x1_class_results);
x2_target_error_4 = confused(x2_class, x2_class_results);

lr3 = .01;
epochs3 = 1000;
net3 = NN_training(input, target, hidden, lr3, epochs3, train_fun);
net3 = train(net3, input, target);
h = plot_dec_region(net3,[-5,5;5,-5; 5,5; -5,-5]',[min_xy(1),max_xy(1),min_xy(2),max_xy(2)],.01);

x1_target = round(net3(input));
x2_target = round(net3(test));
x1_class_results = x1_target(1,:);
x1_class = target(1,:);
x2_class_results = x2_target(1,:);
x2_class = testtarget(1,:);
x1_target_error_15 = confused(x1_class, x1_class_results);
x2_target_error_15 = confused(x2_class, x2_class_results);

lr4 = .001;
epochs4 = 1000;
net4 = NN_training(input, target, hidden, lr4, epochs4, train_fun);
net4 = train(net4, input, target);
net4 = train(net4, input, target);
net4 = train(net4, input, target);
h = plot_dec_region(net4,[-5,5;5,-5; 5,5; -5,-5]',[min_xy(1),max_xy(1),min_xy(2),max_xy(2)],.01);

x1_target = round(net4(input));
x2_target = round(net4(test));
x1_class_results = x1_target(1,:);
x1_class = target(1,:);
x2_class_results = x2_target(1,:);
x2_class = testtarget(1,:);
x1_target_error_15_2 = confused(x1_class, x1_class_results);
x2_target_error_15_2 = confused(x2_class, x2_class_results);

nodes2x1 =   sprintf('lr = .01  epochs = 300   Error train  :  %0.2f%%',x1_target_error_2*100);
nodes2x2 =   sprintf('lr = .01  epochs = 300   Error test  :  %0.2f%%',x2_target_error_2*100);
nodes4x1 =   sprintf('lr = .001 epochs = 300   Error train  :  %0.2f%%',x1_target_error_4*100);
nodes4x2 =   sprintf('lr = .001 epochs = 300   Error test  :  %0.2f%%',x2_target_error_4*100);
nodes15x1 =  sprintf('lr = .01  epochs = 1000  Error train  :  %0.2f%%',x1_target_error_15*100);
nodes15x2 =  sprintf('lr = .01  epochs = 1000  Error test  :  %0.2f%%',x2_target_error_15*100);
nodes15x12 = sprintf('lr = .001 epochs = 1000  Error train  :  %0.2f%%',x1_target_error_15_2*100);
nodes15x22 = sprintf('lr = .001 epochs = 1000  Error test  :  %0.2f%%',x2_target_error_15_2*100);

disp(nodes2x1);disp(nodes2x2);
disp(nodes4x1);disp(nodes4x2);
disp(nodes15x1);disp(nodes15x2);
disp(nodes15x12);disp(nodes15x22);