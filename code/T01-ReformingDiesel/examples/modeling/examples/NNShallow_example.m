ndata = 600;
x = 2*(rand(ndata,1)-0.5);
y = 2*(rand(ndata,1)-0.5);
%
z = sin(2*pi*(x.^2 + y.^2));

%
net = feedforwardnet([5]);
net = cascadeforwardnet([25 25 5 5]);

inputs = [x' ;y'];
targets = z';
[net,tr] = train(net,inputs,targets);

%
%%
xline = linspace(-1.5,1.5,100);
yline = linspace(-1.5,1.5,100);

[xms,yms] = meshgrid(xline,yline);

zpred = net([xms(:) yms(:)]');
zpred = reshape(zpred,100,100)';
%%


%%
%%
figure(2)
clf
plot3(x,y,z,'.','MarkerSize',20,'Color','r');
grid on 
hold on
is = surf(xms,yms,zpred,'FaceAlpha',0.5);
shading interp
legend(is,'Pred')