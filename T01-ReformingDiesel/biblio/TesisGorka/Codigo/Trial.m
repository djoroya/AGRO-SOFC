clear 
load('/Users/djoroya/Documents/GitHub/AGRO-SOFC/biblio/TesisGorka/Codigo/Model&Optimization/data/data01.mat')

inputs = con2seq(data_train.input);
targets = con2seq(data_train.output);
tspan  = data_train.tspan;
%
d1 = [1:4];
d2 = [1:4];

narx_net = narxnet(d1,d2,10);
%narx_net = closeloop(narx_net)

narx_net.divideFcn = '';
narx_net.trainParam.min_grad = 1e-10;
[p,Pi,Ai,t] = preparets(narx_net,inputs,{},targets);
%
narx_net = train(narx_net,p,t,Pi);
%
yp = sim(narx_net,p,Pi);
e = cell2mat(yp)-cell2mat(t);
plot(e')

%%


inputs_trial = con2seq(data_val.input);
targets_trial = con2seq(data_val.output);

[p,Pi,Ai,t] = preparets(narx_net,inputs_trial,{},targets_trial);
yp = sim(narx_net,p,Pi);
figure(1)
clf
subplot(4,1,1)
plot(data_val.input')
title('input')


subplot(4,1,2)
plot(data_val.output')
title('output real')

subplot(4,1,3)
plot(cell2mat(yp)')
title('output pred')



subplot(4,1,4)
plot(abs(data_val.output' - cell2mat(yp)'))
title('error')


