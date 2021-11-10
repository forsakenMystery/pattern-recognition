m=[ 1 8 13; 1 6 1];
S(:,:,1) = 6 * [1 0;0 1];
S(:,:,2) = 6 * [1 0;0 1];
S(:,:,3) = 6 * [1 0;0 1];
P = [1.0/3 1.0/3 1.0/3]';
N = 1000;
[X,y] = generate_gauss_classes(m, S, P, N);
ykn = k_nn_classifier(m,[1 2 3], 1, X);
compute_error(y, ykn)
ykn = k_nn_classifier(m,[1 2 3], 3, X);
compute_error(y, ykn)
% I draw a conclusion when you use bigger number of class you get better
% result at the same time if you use too big of a class you get nothing
% more