function h = plot_dec_region( nn, u, limits, dx, input, target )
[xg,yg] = meshgrid(limits(1):dx:limits(2),limits(3):dx:limits(4));
xg = reshape(xg,1,numel(xg));
yg = reshape(yg,1,numel(yg));
output = nn([xg; yg]);
output = round(output);
omega1=(output(1,:)==1);
omega2=(output(1,:)==-1);
h = figure;
axis equal; 
hold on;
plot(xg(omega1),yg(omega1),'r.');
plot(xg(omega2),yg(omega2),'b.');
plot(u(1,1:2),u(2,1:2),'k+','markersize',10);
plot(u(1,3:4),u(2,3:4),'ko','markersize',10);
