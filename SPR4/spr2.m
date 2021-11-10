rand('seed',0);
randn('seed',0);

mu1 = [ 1, 1 ].';
mu2 = [ 0, 0 ].';
sigmasSquared = 0.2; 
d = size(mu1,1); 

nFeatsBig = 500; 
nFeats = 50;

X1 = mvnrnd( mu1, sigmasSquared*eye(d), nFeatsBig ); 
X1( find( X1(:,1) + X1(:,2) < 1 ), : ) = [];
X1 = X1( 1:nFeats, : ); 

X2 = mvnrnd( mu2, sigmasSquared*eye(d), nFeatsBig ); 
X2( find( X2(:,1) + X2(:,2) > 1 ), : ) = [];
X2 = X2( 1:nFeats, : ); 

h1 = plot( X1(:,1), X1(:,2), '.b' ); hold on; 
h2 = plot( X2(:,1), X2(:,2), '.r' ); hold on; 
legend( [h1,h2], {'class 1', 'class 2'} ); 

data = [X1;X2];
inds_1 = 1:nFeats;
inds_2 = (nFeats+1):(2*nFeats);

delta_x = [ -1*ones(nFeats,1); +1*ones(nFeats,1) ];

rho = 0.7; 
maxIters = 1000;

data = [ data, ones(size(data,1),1) ]; 
l_extended = size(data,2);

w_i = randn(size(data,2),1); 
Niters = 0; 
while( 1 )

  predicted_class = data * w_i;
  predicted_class(inds_2) = -predicted_class(inds_2);
  Y = find( predicted_class < 0 ); 
  if( isempty(Y) ) 
    break;
  end
  
  delta_w = sum( data( Y, : ) .* repmat( delta_x( Y ), 1, l_extended ), 1 ).';

  w_i = w_i - rho * delta_w;
  
  Niters = Niters + 1; 
  if( Niters > maxIters ) 
    fprintf('max number of iterations= %10d exceeded\n',maxIters);
    break
  end
end

x1_grid = linspace( min(data(:,1)), max(data(:,1)), 50 );
x2_db   = ( -w_i(3) - w_i(1) * x1_grid ) / w_i(2);
plot( x1_grid, x2_db, '-g' );
title('perceptron computed decision line');