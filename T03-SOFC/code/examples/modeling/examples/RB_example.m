ndata = 300;
x = rand(ndata,1)-0.5;
y = rand(ndata,1)-0.5;
%
z = x.^2 + y.^2;
%%
figure(1)
clf
plot3(x,y,z,'.','MarkerSize',20,'Color','r');
grid on 
hold on
%
eg = 0.5; % sum-squared error goal
sc = 1;    % spread constant
net = newrb([x y]',z',eg,sc);
%

xline = linspace(-0.5,0.5,100);
yline = linspace(-0.5,0.5,100);

[xms,yms] = meshgrid(xline,yline);

zpred = net([xms(:) yms(:)]');
zpred = reshape(zpred,100,100);
is = surf(xms,yms,zpred,'FaceAlpha',0.5);
shading interp
