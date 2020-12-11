clear
[x,t] = maglev_dataset;
%%
figure(1)
clf
subplot(2,2,1)
plot([x{:}],[t{:}],'.')
%%
subplot(2,2,2)
plot([x{:}],'-')

subplot(2,2,3)
plot([t{:}],'-')
%%
net = narxnet(1:2,1:2,10);
[Xs,Xi,Ai,Ts] = preparets(net,x,{},t);
%%
subplot(2,2,4)
hold on
plot(reshape([Xs{:}],2,3999)')

plot(reshape([Xi{:}],2,3999)')

%%
subplot(2,2,4)
plot(reshape([Ts{:}],2,3999)')
%%
[net,tr] = train(net,Xs,Ts,Xi,Ai);
%%

