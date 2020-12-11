clear

load('/Users/djoroya/Documents/GitHub/AGRO-SOFC/code/Model&Optimization/data/expdata/reformer-data.mat')
figure(1)
clf
take = randsample(1:10001,500);
%[inputs,outputs] = scaleSOFC(inputs,outputs);


%%


%%
views = {[0,90],[0,0],[90,0],3};
for i=1:4
subplot(2,2,i)
scatter3(inputs.RatioSteamC(take),inputs.RatioHC(take),inputs.Temperature(take),[], ...
         inputs.RatioOC(take));
     
title('Inputs')
xlabel('S/C')
ylabel('H/C')
zlabel('Temperature')
colormap jet 
c = colorbar;
c.Label.String = 'O/C';
view(views{i})
end
grid on

figure(2)
clf

scatter3(outputs.H2(take),outputs.CO(take),inputs.RatioHC(take),[],inputs.Temperature(take));
colormap jet 
c = colorbar;
c.Label.String = 'Temperature';

xlabel('H2')
ylabel('CO')

%%

[b,bint,r,rint,stats] = regress(outputs.CO,table2array(inputs));

%%
names = horzcat(inputs.Properties.VariableNames ,outputs.Properties.VariableNames(1));
mdl = fitlm(table2array(inputs),outputs.CO,'VarNames',names);

%%
names = horzcat(inputs.Properties.VariableNames ,outputs.Properties.VariableNames(2));
mdl = fitlm(table2array(inputs),outputs.CO,'VarNames',names);

%%
figure(3);
clf
subplot(3,2,1)
scatter3(inputs.RatioHC,inputs.RatioOC,outputs.H2,[],inputs.Temperature)
title('Out: H2')
colorbar
xlabel('H/C')
ylabel('O/C')
grid on 
%%
subplot(3,2,2)
scatter3(inputs.RatioHC,inputs.RatioOC,outputs.CO,[],inputs.Temperature)
title('Out: C0')
colorbar

xlabel('H/C')
ylabel('O/C')
grid on 
%%
subplot(3,2,3)
scatter3(inputs.RatioHC,inputs.RatioSteamC,outputs.H2,[],inputs.Temperature)
title('Out: H2')
colorbar
xlabel('H/C')
ylabel('S/C')
grid on 


%%
subplot(3,2,4)
scatter3(inputs.RatioHC,inputs.RatioSteamC,outputs.CO,[],inputs.Temperature)
title('Out: C0')
colorbar

xlabel('H/C')
ylabel('S/C')
grid on 
%%
subplot(3,2,5)
scatter3(inputs.RatioOC,inputs.RatioSteamC,outputs.H2,[],inputs.Temperature)
title('Out: H2')
colorbar
xlabel('O/C')
ylabel('S/C')
grid on 
%%
subplot(3,2,6)
scatter3(inputs.RatioOC,inputs.RatioSteamC,outputs.CO,[],inputs.Temperature)
title('Out: C0')
colorbar

xlabel('O/C')
ylabel('S/C')
grid on 