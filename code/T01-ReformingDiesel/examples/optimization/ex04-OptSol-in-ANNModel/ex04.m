clear 

A = [ -2.0  0.5  1.20  ;
       0.0 -1.0  0.10  ;
      -0.2  0.1 -1.00 ];
 
B = [ 1 0 ; ...
      0 1 ; ...
      0 0 ];
 
C = [1 0 1]';

T = 2; Nt = 100;
tspan = linspace(0,T,Nt);
x0 = [0 0 1]';

[ut_opt,xt_opt,xt_free] = SolveOCP(A,B,C,tspan,x0);
%%
figure(1)
subplot(3,1,1)
plot(ut_opt')
title('ut opt')
subplot(3,1,2)
plot(xt_opt')
title('xt opt')
subplot(3,1,3)
plot(xt_free')
title('xt Free')

%%
[ds_train,ds_test] = GenDataLinearSys(A,B,C,tspan);
ds_opt = ds(ut_opt,xt_opt(:,1:end-1));
%%

neurons = [100 100];
net = feedforwardnet(neurons);

no = 4;

[inputs_narx,output_narx] = serveral_narx(ds_train,no);

net.trainFcn='trainbfg';
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-5;
net.performParam.regularization = 0.01;

net= train(net,inputs_narx,output_narx);
%%
testplot(ds_train(3),net,no)
%%
testplot(ds_test(1),net,no)


%%
testplot(ds_opt,net,no)
