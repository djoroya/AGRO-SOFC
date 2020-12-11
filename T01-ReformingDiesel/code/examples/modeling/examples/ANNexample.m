clear 
load('/Users/djoroya/Documents/GitHub/AGRO-SOFC/code/Model&Optimization/data/expdata/reformer-data.mat')

%% Scale data

Ni = 2;
No = 1;
inputs = array2table(3*(rand(500,Ni)-0.5));
outputs = array2table(sum(inputs{:,:}.^2,2));

Ndata = 400;
indtake = randsample(500,Ndata,false);
%% Split data in test and train
train_in = inputs(indtake(1:floor(Ndata/2)),:);
test_in  = inputs(indtake(1+floor(Ndata/2):end),:);
%
train_out = outputs(indtake(1:floor(Ndata/2)),:);
test_out  = outputs(indtake(1+floor(Ndata/2):end),:);
%% Normalization
%
layers = [
    imageInputLayer([Ni 1 1])
    
    fullyConnectedLayer(10)
    reluLayer
    
    fullyConnectedLayer(No)
    
    
    regressionLayer];

%%
miniBatchSize = 400;
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',5000, ...
    'InitialLearnRate',1e-1, ...
    'LearnRateDropFactor',0.9, ...
    'Shuffle','every-epoch', ...
    'Plots','none', ...
    'Verbose',true);

options.ExecutionEnvironment = 'parallel'
%%
net = trainNetwork(reshape(train_in{:,:}  ,Ni,1,1,Ndata/2), ...
                   reshape(train_out{:,:} ,1,1,No,Ndata/2), ...
                   layers,options);
               
%%
clf
plot3(inputs{:,1},inputs{:,2},outputs{:,1},'*')

xline = linspace(-3,3,200);
yline = linspace(-3,3,200);

[xms,yms] = meshgrid(xline,yline);

Vpred = predict(net,reshape([xms(:)';yms(:)'],2,1,1,200*200));
Vpred = reshape(Vpred,200,200)';
hold on

surf(xms,yms,Vpred)
shading interp