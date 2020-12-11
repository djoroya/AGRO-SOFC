function plotAllDataSOFC(DataSOFCtable)
inds = 3:57;

iter = 0;
for i = 1:length(inds)

    iter = iter +1;
    subplot(8,7,iter)

    plot(DataSOFCtable.Time,DataSOFCtable{:,inds(i)})
    xlabel('')
    if i ~= 1
    xticks([])
    end
    title("D"+inds(i)+" | "+DataSOFCtable.Properties.VariableNames{inds(i)},'Interpreter','latex')
end
end

