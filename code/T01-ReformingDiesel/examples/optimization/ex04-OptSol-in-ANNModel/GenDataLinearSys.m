function [ds_train,ds_test] = GenDataLinearSys(A,B,C,tspan)

[N,~]=size(A);
for iter = 1:10
%%
% definimos los inputs 

cf = @(j) rand*sin(0.5*pi*j*tspan) + 2*rand*cos(0.5*pi*j*tspan);

ci = rand(10,1)-0.5;
udata = 10*[sum(ci.*cf(randi(3,10,1))); ...
         sum(ci.*cf(randi(3,10,1)))] + 0.5*(0.5-rand(size(tspan)));
%udata = 1+0.1*[sin(pi*tspan) ;cos(pi*tspan)] + 0.05*(0.5-rand(size(tspan)));
%%
ut = @(t) interp1(tspan,udata',t)';

x0 = 0.5 - rand(N,1);
[~,xdata] = ode45(@(t,x) A*x+B*ut(t) + C,tspan,x0);

ds_train(iter) = ds(udata,xdata');


%%
udata_val = 0.2*[sin(0.4*pi*tspan) ;cos(0.2*pi*tspan)];
ut = @(t) interp1(tspan,udata_val',t)';

x0 = 0.5 - rand(N,1);
[~,xdata_val] = ode45(@(t,x) A*x+B*ut(t),tspan,x0);

ds_test(iter) = ds(udata_val,xdata_val');

end

end

