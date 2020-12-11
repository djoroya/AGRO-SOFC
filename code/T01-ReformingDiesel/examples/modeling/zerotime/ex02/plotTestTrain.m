function plotTestTrain(net,inputs,outputs,out)


%     Inputs
%    {'Temperature'}    {'RatioSteamC'}    {'RatioOC'}    {'RatioHC'}
%     Outputs
%     {'CO'}    {'H2'}


predict_outputs = net(inputs{:,:}')';

[ndata,~] = size(inputs);

err = perform(net,outputs{:,:},predict_outputs);
err = sqrt(err);

subplot(2,2,1)

scatter3(inputs.RatioHC     , ...
         inputs.RatioSteamC , ...
         outputs.(out)         , ...
         []                 , ...
         inputs.Temperature)
%
xlabel('H/C')
ylabel('S/C')
zlabel(out)
colormap(jet)
ic = colorbar;
ic.Label.String = 'Temperature';

title("real "+out+"  |  ndata = "+ndata+ " | perform = "+num2str(err,'%.2e'))
%zlim([0 0.8])

%%
outindex = arrayfun(@(i) strcmp(outputs.Properties.VariableNames{i},out),1:2);
subplot(2,2,2)
scatter3(inputs.RatioHC         , ...
         inputs.RatioSteamC     , ...
         predict_outputs(:,outindex)   , ...
         []                     , ...
         inputs.Temperature)

xlabel('H/C')
ylabel('S/C')
zlabel(out)


title("pred "+out+"  |  ndata = "+ndata+ " | perform = "+num2str(err,'%.2e'))
%zlim([0 0.8])
colormap(jet)
ic = colorbar;
ic.Label.String = 'Temperature';

%%
subplot(2,2,3)

scatter3(inputs.RatioHC     , ...
         inputs.RatioOC , ...
         outputs.(out)         , ...
         []                 , ...
         inputs.Temperature)
%
xlabel('H/C')
ylabel('O/C')
zlabel(out)
colormap(jet)
ic = colorbar;
ic.Label.String = 'Temperature';

title("real "+out+"  |  ndata = "+ndata+ " | perform = "+num2str(err,'%.2e'))
%zlim([0 0.8])

%%
outindex = arrayfun(@(i) strcmp(outputs.Properties.VariableNames{i},out),1:2);
subplot(2,2,4)
scatter3(inputs.RatioHC         , ...
         inputs.RatioOC     , ...
         predict_outputs(:,outindex)   , ...
         []                     , ...
         inputs.Temperature)

xlabel('H/C')
ylabel('O/C')
zlabel(out)


title("pred "+out+"  |  ndata = "+ndata+ " | perform = "+num2str(err,'%.2e'))
%zlim([0 0.8])
colormap(jet)
ic = colorbar;
ic.Label.String = 'Temperature';
end

