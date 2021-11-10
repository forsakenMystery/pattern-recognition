X1 = [ 0, 0; 0, 1 ];
X2 = [ 1, 0; 1, 1 ]; 
h1 = plot( X1(:,1), X1(:,2), '.b' ); hold on; 
h2 = plot( X2(:,1), X2(:,2), '.r' ); hold on; 
legend( [h1,h2], {'class 1', 'class 2'} ); axis( [-1,+2,-1,+2] ); 

nFeats = 2; 
data = [X1;X2];
inds_1 = 1:nFeats;
inds_2 = (nFeats+1):(2*nFeats);

data = [ data, ones(size(data,1),1) ]; 
l_extended = size(data,2);

data(inds_2,:) = -data(inds_2,:);

rho = 1.0;
maxIters = 1000;

w_i = [0,0,0].';

Niters = 0; 
fprintf('w_i=[%10.1f, %10.1f, %10.1f] \n', w_i(1),w_i(2),w_i(3) );
while( 1 )

  number_of_errors = 0;
  for ii=1:size(data,1)
    asghar=size(data, 1)
    mamad = data(ii, :)
    
    if( data(ii,:) * w_i <= 0 ) % this sample is misclassified
      number_of_errors = number_of_errors + 1; 
      w_i = w_i + rho * (data(ii,:).');
      fprintf('w_i=[%10.1f, %10.1f, %10.1f] \n', w_i(1),w_i(2),w_i(3) );
    end
  end
  if( number_of_errors==0 )
    break;
  end
  
  Niters = Niters + 1; 
  if( Niters > maxIters ) 
    fprintf('max number of iterations= %10d exceeded\n',maxIters);
    break
  end
end

addpath('~/Matlab');
vline(1/2,'-r');
title('reward-punishment perceptron computed decision line');