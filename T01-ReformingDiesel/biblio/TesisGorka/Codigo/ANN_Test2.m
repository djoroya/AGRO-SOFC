function ANN_Test2

clear all;
close all;
clc;

mod=3;
neuronas=4;
trnfcn=3;
MAPE = 100;
cont=1;

excelfile=strcat(pwd,'\Results\ANNRealDataTestRESUMEN.xlsx');
exceldata={'cont','MAPE','RMSE'};
xlswrite(excelfile,exceldata,1,'B1');
xlswrite(excelfile,{strcat('Mod ',int2str(mod))},1,'A1');

% while MAPE > 15.0
for cont=1:15
    [RMSE,MAPE]=ANN_FF_RealData2(mod,neuronas,trnfcn,cont);
    fprintf('ItGbl: %d - MAPE: %.4f, RMSE: %.4f',cont,MAPE,RMSE);
%     cont=cont+1;
    xlsrange=strcat('B',num2str(cont+1));
    xlswrite(excelfile,[cont MAPE RMSE],1,xlsrange);  
end

end