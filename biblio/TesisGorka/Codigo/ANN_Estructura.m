function ANN_Estructura(iter,mod,neur)%Trainfcn
% function ANN_Estructura(iter,mod)%Neuronas
% clc;
% close all;
% clear all;
folder = strcat('ResultsMod',int2str(mod));
mkdir(folder);

excelfile=strcat(pwd,'/',folder,'/ANNRealDataLOOCV_EstrcRESUMEN.xlsx');
% exceldata={'Neuronas','mean(MAPE)','mean(RMSE)','min(MAPE)','min(RMSE)'};
% xlswrite(excelfile,exceldata,1,'B1');
% xlswrite(excelfile,{strcat('Mod ',int2str(1))},1,'A1');
resultados=[];

MAPExN=[];
RMSExN=[];
for j=1:17 %Neuronas (1-25) o trainfcn (1-17)
    MAPExI=[];
    RMSExI=[];
    for i=1:iter
%         [MAPECV,RMSECV]=ANN_FF_RealData_LOOCV(j,i,mod,folder);%NO SE USA
%         [MAPECV,RMSECV]=ANN_FF_RealData_LOOCV(j,i,mod,folder,4);%Neuronas
        [MAPECV,RMSECV]=ANN_FF_RealData_LOOCV(neur,i,mod,folder,j);%trainfcn
        close all;
        fprintf('\nMod: %d - Neuronas: %d - Iteración: %d, MAPECV = %f,  RMSE = %f\n',mod,j,i,MAPECV,RMSECV);
        MAPExI(i)=MAPECV;
        RMSExI(i)=RMSECV;
    end
    MAPExN(j)=mean(MAPExI);
    RMSExN(j)=mean(RMSExI);
    fprintf('\nMod: %d - Neuronas: %d, MAPExN(mejor) = %f,  RMSExN(mejor) = %f',mod,j,min(MAPExI),min(RMSExI));
    fprintf('\nMod: %d - Neuronas: %d, MAPExN(media) = %f,  RMSExN(media) = %f\n\n',mod,j,MAPExN(j),RMSExN(j));

%     xlsrange=strcat('B',num2str(j+1));
%     xlswrite(excelfile,[j MAPExN(j) RMSExN(j) min(MAPExI) min(RMSExI)],1,xlsrange); 
    resultados = [resultados; mod, j, MAPExN(j), RMSExN(j), min(MAPExI), min(RMSExI)];
end
csvwrite(excelfile,resultados);
resultados
