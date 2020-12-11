clear 

pathfolder = [pwd,'/data/19_Data_Ainhoa_SOFC_SingleCell_ESC/SPS_Data/'];
files = dir(pathfolder);

%%0 
path = pwd; 
%%

figure('Unit','norm','pos',[0 0 1 1])
for i = 3:10
    filename = files(i).name;

    DataSOFCtable  = DataSOFC(filename);
    plotAllDataSOFC(DataSOFCtable)
    print('-dpng',fullfile(pathfolder,'img','all',num2str(i)+".png"))
end

%%
%%
figure(1)
clf
figure('Unit','norm','pos',[0 0 1 1])
for i = 3:50
    filename = files(i).name;

    DataSOFCtable  = DataSOFC(filename);
    plotnonzeroDataSOFC(DataSOFCtable)
    print('-dpng',fullfile(pathfolder,'img','nonzero',num2str(i)+".png"))
end


%%
figure(1)
clf

inds = [3:20 26 32 44 50:54 57];

for i = 1:length(inds)
    subplot(3,9,i)
    plot(DataSOFCtable.Time,DataSOFCtable{:,inds(i)})
    title("D"+inds(i)+" | "+DataSOFCtable.Properties.VariableNames{inds(i)},'Interpreter','latex')
end


%%
figure(2)
clf
iter = 0;
for i = [5 50 51 26 28]
    iter = iter + 1;
    subplot(2,3,iter)
    plot(DataSOFCtable.Time,DataSOFCtable{:,i})
    title("D"+i+" | "+DataSOFCtable.Properties.VariableNames{i},'Interpreter','latex')
end