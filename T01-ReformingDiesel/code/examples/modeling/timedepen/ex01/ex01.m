clear 

load('data/data01');

%%
nid = 1; % number of inputs delay


%%

neurons = [20 20];
net = feedforwardnet(neurons);

no = 2;

[inputs_narx,output_narx] = serveral_narx(ds_train,no);

net.trainFcn='trainbfg';
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-5;
net.performParam.regularization = 0.001;

net= train(net,inputs_narx,output_narx);


%%
%
ifig  = figure(1);
ifig.Name = 'train dataset';
clf
testplot(ds_train(1),net,no)
%
ifig  = figure(2);

ifig.Name = 'test dataset';
clf
testplot(ds_test(1),net,no)
