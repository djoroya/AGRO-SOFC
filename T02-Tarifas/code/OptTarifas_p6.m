%%
clear 

%
% orden de indices  
% i - power limits 
% j - part
% m - month
% l - trial

k= [1, 0.5, 0.37, 0.37, 0.37, 0.17]';
k = reshape(k,6,1,1,1);

coeff = 1.4064;

c = (6:-1:1)' + rand(6,1);
c = reshape(c,6,1,1,1);

%%
nquarter = 10;
pi = 1500*rand(6,nquarter,12);
pi = normrnd(500,200,[6 nquarter 12]);

theta = @(x) 0.5 + 0.5*tanh(x);

sq = @(p) sqrt( ...
                   sum(  theta( pi - p ).*(pi - p).^2 , 2 )    ...
                );

phi = @(p) sum(p.*c + coeff*k.*sq(p),1);

cost = @(p) sum(phi(p),3);
%%
p0 = 1000*rand(6,1);
p0(p0<450) = 450;
p0 = sort(p0);
%% GA
clf 
%
hold on 
%
npop = 1000;
population = 10000*rand(6,1,1,npop);

population(population<450) = 450;
population = sort(population);

%
color = jet(100);
for i = 1:100
    % crossover
    childrens = crossover(population,300);
    population = cat(4,population,childrens);
    % mutation
    ind_mu = randsample(1:(npop+300),50,true);
    population(:,:,:,ind_mu) = population(:,:,:,ind_mu) + 10*rand(1,1,1,50);
    population(population<450) = 450;

    % selection
    cost_values = cost(population);
    [~,ind] = sort(cost_values(:));
    population = population(:,:,:,ind(1:npop));
    %
    population = population(:,:,:,randsample(1:npop,npop,false));
    % plot
    if mod(i,10) == 1
        plot(cost_values(:),'color',color(i,:))
    end
end 

cost(population(:,:,:,1))

function childrens = crossover(population,n)
    [~,~,~,npop] = size(population);
    
    childrens = zeros(6,1,1,n);
    for i = 1:n
        ind = randsample(1:npop,2,true);
        childrens(:,:,:,i) = mean(population(:,:,:,ind),4);
    end
end