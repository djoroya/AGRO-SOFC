clear 
N = 4;
A = -1*eye(N,N)+0.04*(0.5-rand(N,N)) ;

A(end,:) = 0.1*A(end,:);
M = 1;
B = rand(N,M);

C = 5*(rand(N,1)-0.5);
tspan = linspace(0,10,500);

%%
% definimos los inputs 

cf = @(j) rand*sin(0.5*pi*j*tspan) + 2*rand*cos(0.5*pi*j*tspan);

ci = rand(10,1)-0.5;
switch M
    case 1
        udata = [sum(ci.*cf(randi(3,10,1)))];
    case 2
        udata = [sum(ci.*cf(randi(3,10,1))); ...
                 sum(ci.*cf(randi(3,10,1)))] + 0.5*(0.5-rand(size(tspan)));
end
%%
ut = @(t) interp1(tspan,udata',t)';

x0 = 0.5 - rand(N,1);
[~,xdata] = ode45(@(t,x) A*x+B*ut(t) + C,tspan,x0);

ds_train = ds(udata,xdata');


%%

switch M
    case 1
        udata_val = 0.2*[sin(0.4*pi*tspan)];
    case 2
        udata_val = 0.2*[sin(0.4*pi*tspan) ;cos(0.2*pi*tspan)];

end

ut = @(t) interp1(tspan,udata',t)';

x0 = 0.5 - rand(N,1);
[~,xdata_val] = ode45(@(t,x) A*x+B*ut(t),tspan,x0);

ds_test = ds(udata_val,xdata_val');

save('data/simdata/data01.mat','ds_train','ds_test')
