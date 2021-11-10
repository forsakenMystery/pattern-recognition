function [px]=Parzen_gauss_kernel(X,h,xleftlimit,xrightlimit)
[l,N]=size(X);
xstep=h;
k=1;
x=xleftlimit;
while x<xrightlimit+xstep/2
    px(k)=0;
    for i=1:N
        xi=X(:,i);
        px(k)=px(k)+exp(-(x-xi)'*(x-xi)/(2*h^2));
    end
    px(k)=px(k)*(1/N)*(1/(((2*pi)^(l/2))*(h^l)));
    k=k+1;
    x=x+xstep;
end