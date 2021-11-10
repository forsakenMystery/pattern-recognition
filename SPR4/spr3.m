% Generate the required data:
%
mu1 = [ 1, 1 ].';
mu2 = [ 0, 0 ].';
sigmasSquared = 0.2; 
d = size(mu1,1); 

nFeats = 100;

X1 = mvnrnd( mu1, sigmasSquared*eye(d), nFeats ); 
X2 = mvnrnd( mu2, sigmasSquared*eye(d), nFeats ); 

h1 = plot( X1(:,1), X1(:,2), '.b' ); hold on; 
h2 = plot( X2(:,1), X2(:,2), '.r' ); hold on; 
legend( [h1,h2], {'class 1', 'class 2'} ); 

data = [X1;X2];
inds_1 = 1:nFeats;
inds_2 = (nFeats+1):(2*nFeats);

N = size(data,1);

Y_targets = +1*ones(2*nFeats,1);
Y_targets(inds_2) = -1; 


rho = 0.01;


maxIters = 500;

% Append +1 to all data: 
data = [ data, ones(size(data,1),1) ]; 
l_extended = size(data,2);

w_i = randn(size(data,2),1);
solution_found = 0; 
for iter=1:maxIters
  
  for ii=1:N
    x = data(ii,:).';
    y = Y_targets(ii);
    delta_w = rho * x * ( y - x' * w_i );
    y - x' * w_i
    x
    x * ( y - x' * w_i )
    if( max(abs(delta_w)) < 1.e-6 ) 
      solution_found = 1;
      break;
    end
    w_i = w_i + delta_w; 
  end
  
  if( solution_found ) 
    fprintf('solution found after iter= %10d iterations\n',iter);
    break;
  end
end

x1_grid = linspace( min(data(:,1)), max(data(:,1)), 50 );
x2_db   = ( -w_i(3) - w_i(1) * x1_grid ) / w_i(2);
plot( x1_grid, x2_db, '-g' );
title('LMS computed decision line');