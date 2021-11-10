function [X,y]=generate_gauss_classes(m,S,P,N)
    [l,c]=size(m);
    X=[];
    y=[];
    for j=1:c
        t=mvnrnd(m(:,j),S(:,:,j),fix(P(j)*N));
        X=[X t'];
        y=[y ones(1,fix(P(j)*N))*j];
    end
end
