clear 

load('data/data03');

%%

neurons = [20 20];
net = feedforwardnet(neurons);

no = 4;

[inputs_narx,output_narx] = serveral_narx(ds_train,no);

net.trainFcn='trainbfg';
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-5;
net.performParam.regularization = 0.01;

net= train(net,inputs_narx,output_narx);


%%
%

for iter = 1:3
ifig  = figure(iter);
ifig.Name = 'train dataset';
clf
testplot(ds_test(iter),net,no)
end

for iter = 4:6
ifig  = figure(iter);
ifig.Name = 'test dataset';
clf
testplot(ds_test(iter),net,no)
end