clear 
N = 4;
A = -1*eye(N,N)+0.1*(0.5-rand(N,N)) ;

M = 2;
B = zeros(N,M);
B(1,1) = 1;
B(2,2) = 1;

B(end,end) = 1;

C = 0.5*(rand(N,1));
tspan = linspace(0,10,500);

%%
% definimos los inputs 

for iter = 1:10
cf = @(j) rand*sin(0.5*pi*j*tspan) + 2*rand*cos(0.5*pi*j*tspan);

ci = rand(10,1)-0.5;
udata = [sum(ci.*cf(randi(4,10,1))); ...
         sum(ci.*cf(randi(4,10,1)))];
%udata = 1+0.1*[sin(pi*tspan) ;cos(pi*tspan)] + 0.05*(0.5-rand(size(tspan)));
%%
ut = @(t) interp1(tspan,udata',t)';

x0 = 0.5 - rand(N,1);
[~,xdata] = ode45(@(t,x) A*x+B*ut(t) + C,tspan,x0);

ds_train(iter) = ds(udata,xdata');


%%
udata_val = [sum(ci.*cf(randi(4,10,1))); ...
         sum(ci.*cf(randi(4,10,1)))] ;
     
ut = @(t) interp1(tspan,udata_val',t)';

x0 = 0.5 - rand(N,1);
[~,xdata_val] = ode45(@(t,x) A*x+B*ut(t),tspan,x0);

ds_test(iter) = ds(udata_val,xdata_val');
end

save("data/simdata/data02",'ds_train','ds_test','A','B','C')
