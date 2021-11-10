function errors=confused(y,y_est) 
[q, N]=size(y);
c=max(y);
errors=0; 
for i=1:N
    if(y(i)~=y_est(i)) 
        errors=errors+1;
    end
end

errors=errors/N;