clear 
load('data/19_Data_Ainhoa_SOFC_SingleCell_ESC/DataSOFC.mat')
%%
figure(1)
clf

inds = 3:57;

iter = 0;
for i = 1:57
    if length(unique(DataSOFCtable{:,inds(i)}) ) == 1
        continue
    end
    iter = iter +1;
    subplot(5,5,iter)

    plot(DataSOFCtable.Time,DataSOFCtable{:,inds(i)})
    xlabel('')
    xticks([])
    title("D"+inds(i)+" | "+DataSOFCtable.Properties.VariableNames{inds(i)},'Interpreter','latex')
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
for i = [50 51 26 28]
    iter = iter + 1;
    subplot(2,2,iter)
    plot(DataSOFCtable.Time,DataSOFCtable{:,i})
    title("D"+i+" | "+DataSOFCtable.Properties.VariableNames{i},'Interpreter','latex')
end