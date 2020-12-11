clear 
load('/Users/djoroya/Documents/GitHub/AGRO-SOFC/code/Model&Optimization/data/expdata/reformer-data.mat')
%% Scale data
%[inputs,outputs] = scaleSOFC(inputs,outputs);

Ndata = 2000;
indtake = randsample(10001,Ndata,false);
%% Split data in test and train
deno = 10;

train_in = inputs(indtake(1:floor(Ndata/deno)),:);
test_in  = inputs(indtake(1+floor(Ndata/deno):end),:);
%
train_out = outputs(indtake(1:floor(Ndata/deno)),:);
test_out  = outputs(indtake(1+floor(Ndata/deno):end),:);
%% Normalization
%               
net = feedforwardnet([10 10 10]);
[net,tr] = train(net,train_in{:,:}',train_out{:,:}');
 

save('/Users/djoroya/Documents/GitHub/AGRO-SOFC/code/Model&Optimization/data/models/ANN-ZeroTime.mat','net')

%%
f = figure(1);
f.Name = 'train-H2';
clf
plotTestTrain(net,train_in,train_out,'H2')
print('-dpng',gcf,'/Users/djoroya/Documents/GitHub/AGRO-SOFC/code/Model&Optimization/examples/modeling/zerotime/ex02/img/TrainH2.png')
%%
f = figure(2);
f.Name = 'train-CO';
clf
plotTestTrain(net,train_in,train_out,'CO')
print('-dpng',gcf,'/Users/djoroya/Documents/GitHub/AGRO-SOFC/code/Model&Optimization/examples/modeling/zerotime/ex02/img/TrainCO.png')

%%
f = figure(3);
f.Name = 'test-H2';
clf
plotTestTrain(net,test_in,test_out,'H2')
print('-dpng',gcf,'/Users/djoroya/Documents/GitHub/AGRO-SOFC/code/Model&Optimization/examples/modeling/zerotime/ex02/img/TestH2.png')

%%
f = figure(4);
f.Name = 'test- CO';
clf
plotTestTrain(net,test_in,test_out,'CO')
print('-dpng',gcf,'/Users/djoroya/Documents/GitHub/AGRO-SOFC/code/Model&Optimization/examples/modeling/zerotime/ex02/img/TestCO.png')
