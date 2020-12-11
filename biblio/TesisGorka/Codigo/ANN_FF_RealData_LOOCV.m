function [MAPEF,RMSEF] = ANN_FF_RealData_LOOCV(neuronas,iter,mod,folder,trfcn)
% function HNN_FF_RealData_LOOCV %pruebas
% function [MAPEF,RMSEF] = HNN_FF_RealData_LOOCV2015norm2_Init2Estructura2%Pruebas
% neuronas=7;%Pruebas
% iter=1;%Pruebas
%% Enero 2015 Validación Leave-One-Out Cross-Validation (LOOCV)
% Fecha: 22/01/2014
% Nuevo Modelo neuronal híbrido con actividad del modelo de conocimiento en
%   el entrenemiento y calculada online mediante la ec. de desactivación. 
% Entradas XLS: W(1), T(2), tao(3), dt(4), XoR(6)
% Salidas XLS: XoR(6)
% Recogida de los datos en bruto y generación automática de los vectores de
%   entrenamiento y validación.
% Validación LOOCV: Entrenamiento del modelo con todos los experimentos
%   menos 1 y validación con ese experimento excluido del entrenamiento. Se
%   repite el proceso de entrenamiento-validación con todos los 
%   experimentos.
% clear all;
% close all;
% clc;

%% Recogida de los datos del proceso BTO
datos=xlsread('Datasets/ExpData_BTOAT10.xlsx',2,'F2:K389');%Exp: 1-23
% datos=xlsread('Datasets/ExpData_BTOAT10_2.xlsx',2,'F2:K518');%Exp: 1-23 + 2 Val
nVal=0;
% datos=xlsread('Datasets\Datos_BTOmod1tiempo-SIMULACION.xlsx',1,'A2:F1153');
% datos=xlsread('Datasets\Datos_BTOmod1tiempo-SIMULACION.xlsx',1,'A2:F1264');%Exp: 1-42 
save(strcat(folder,'/datos_BTO.mat'),'datos');
% fprintf('Datos cargados correctamente\n');
%% Normalización de los datos
maxdat1=max(datos(:,1));
maxdat2=max(datos(:,2));
maxdat3=max(datos(:,3));
% x1_again = mapminmax('reverse',y1,PS)
[datos(:,1),Xn1ps]=mapminmax(datos(:,1)');
[datos(:,2),Xn2ps]=mapminmax(datos(:,2)');
[datos(:,3),Xn3ps]=mapminmax(datos(:,3)');
[datos(:,4),Xn4ps]=mapminmax(datos(:,4)');
% datos(:,1)=datos(:,1)/max(datos(:,1));
% datos(:,2)=datos(:,2)/max(datos(:,2));
% datos(:,3)=datos(:,3)/max(datos(:,3));
% datos(:,4)=datos(:,4)/max(datos(:,4));
save(strcat(folder,'/datos_BTOnorm.mat'),'datos');
% fprintf('Datos normalizados correctamente\n');
%% Preparación de los datos
indices=find(datos(:,5)>=1);
[m,n]=size(indices);
inputs=[];
targets=[];
index=1;
for i=1:m
    if i~=m
        experimento=datos(indices(i):(indices(i+1)-1),:);    
    else
        experimento=datos(indices(i):end,:);
    end    
    nombreexp=strcat (folder,'/Experimento_',int2str(i),'.mat');
    save(nombreexp,'experimento');
    switch mod
        case 1 %ANN 1 - BASE
            inputs=[inputs ; experimento(:,1) experimento(:,2) experimento(:,3) ];
            targets=[targets ; experimento(:,6)];
            if i==1
                index=[index (1+length(experimento))];
            else
                index=[index (index(i)+length(experimento))];
            end
            d=0;
        case 2 %ANN 2 - DYN
            inputs=[inputs ; experimento(1:end-1,1) experimento(2:end,1) experimento(1:end-1,2) experimento(2:end,2) experimento(1:end-1,3) experimento(2:end,3) ];
            targets=[targets ; experimento(2:end,6)];
            if i==1
                index=[index (1+length(experimento)-1)];
            else
                index=[index (index(i)+length(experimento)-1)];
            end
            d=1;
        case 3 %ANN 3 - REC
            inputs=[inputs ; experimento(2:end-1,1) experimento(3:end,1) experimento(2:end-1,2) experimento(3:end,2) experimento(2:end-1,3) experimento(3:end,3) experimento(1:end-2,6) experimento(2:end-1,6)];
            targets=[targets ; experimento(3:end,6)];
            if i==1
                index=[index (1+length(experimento)-2)];
            else
                index=[index (index(i)+length(experimento)-2)];
            end
            d=1;
        case 4 %ANN 4 - TEMP
            inputs=[inputs ; experimento(2:end-1,1) experimento(3:end,1) experimento(2:end-1,2) experimento(3:end,2) experimento(2:end-1,3) experimento(3:end,3) experimento(2:end-1,4) experimento(3:end,4) experimento(1:end-2,6) experimento(2:end-1,6)];
            targets=[targets ; experimento(3:end,6)];
            if i==1
                index=[index (1+length(experimento)-2)];
            else
                index=[index (index(i)+length(experimento)-2)];
            end
            d=1;
        case 5 %ANN 5 - BASEREC n=1 (Minor Revision 2016)
            inputs=[inputs ; experimento(2:end,1) experimento(2:end,2) experimento(2:end,3) experimento(1:end-1,6)];
            targets=[targets ; experimento(2:end,6)];
            if i==1
                index=[index (1+length(experimento)-2)];
            else
                index=[index (index(i)+length(experimento)-2)];
            end
            d=0;
        case 6 %ANN 6 - BASEREC n=2 (Minor Revision 2016)
            inputs=[inputs ; experimento(3:end,1) experimento(3:end,2) experimento(3:end,3) experimento(1:end-2,6) experimento(2:end-1,6)];
            targets=[targets ; experimento(3:end,6)];
            if i==1
                index=[index (1+length(experimento)-2)];
            else
                index=[index (index(i)+length(experimento)-2)];
            end
            d=0;
        case 7 %ANN 7 - BASEREC n=3 (Minor Revision 2016)
            inputs=[inputs ; experimento(4:end,1) experimento(4:end,2) experimento(4:end,3) experimento(1:end-3,6) experimento(2:end-2,6) experimento(3:end-1,6)];
            targets=[targets ; experimento(4:end,6)];
            if i==1
                index=[index (1+length(experimento)-2)];
            else
                index=[index (index(i)+length(experimento)-2)];
            end
            d=0;
    end
            
end
nombreinput=strcat (folder,'/Input.mat');
nombretarget=strcat (folder,'/Target.mat');
save(nombreinput,'inputs');
save(nombretarget,'targets');
save(strcat(folder,'/ExpIndex.mat'),'index');


% fprintf('Datos preparados correctamente\n+-->Se han encontrado %d experimentos\n',m);
%%
% filename=strcat(pwd,'\ResultadosPruebas\ResultadosRNAhibridoRealDataLOOCV.txt');
% fileID=fopen(filename,'w','n','UTF-8');
m=m-nVal;
fprintf('\n+-->Train Experiments: %d - Val. Experiments: %d \n',m,nVal);
for i=1:m   %Experimentos
    %% Entrenamiento del modelo neuronal
%     nombreinput=strcat ('ResultadosPruebas\Input_',int2str(i),'.mat');
%     InputVector=load(nombreinput);
%     nombretarget=strcat ('ResultadosPruebas\Target_',int2str(i),'.mat');
%     TargetVector=load(nombretarget);
%     ep=2000;
%     while ep==2000% || ep<650
%         net = newnarx(InputVector.inputs',TargetVector.targets',0,[1 2],20);
%         In=InputVector.inputs';
%         Tar=TargetVector.targets';
        net = newff(inputs',targets',neuronas);%11
%       net = cascadeforwardnet(60,'trainbr');
        %net = newrb(InputVector.inputs',TargetVector.targets', 0.0,2095.0,600);
        %net = newrb(InputVector.inputs',TargetVector.targets', 0.0,425.0,150); %ok
%         net = newrb(InputVector.inputs',TargetVector.targets', 0.0,1200.0,100);
%         net = newrbe(InputVector.inputs',TargetVector.targets',2.641220272724670e+03);%450.0, 1000.0, 1125, 1140, 1760.0, 2200, 2700, 11670, 10081.438976, 2.641220272724670e+03
%         net = newrbe(InputVector.inputs',TargetVector.targets',2.641220272724670e+03);
        %net = newrb(InputVector.inputs',TargetVector.targets', 0.0,2700.0,200);
%        net.trainFcn='trainbr';
%         net.trainFcn='trainlm';
        switch trfcn
            case 1
                net.trainFcn='trainbr';
                net.trainParam.epochs=1000;
            case 2
                net.trainFcn='trainscg';
                net.trainParam.epochs=1000;
            case 3
                net.trainFcn='trainr';
                net.trainParam.epochs=100;
            case 4
                net.trainFcn='trainlm';
                net.trainParam.epochs=1000;
            case 5
                net.trainFcn='trainc';
                net.trainParam.epochs=100;
            case 6
                net.trainFcn='traincgp';
                net.trainParam.epochs=1000;
            case 7
                net.trainFcn='traingdx';
                net.trainParam.epochs=1000;
            case 8
                net.trainFcn='traincgb';
                net.trainParam.epochs=1000;
            case 9
                net.trainFcn='trainrp';
                net.trainParam.epochs=1000;
            case 10
                net.trainFcn='traincgf';
                net.trainParam.epochs=1000;
            case 11
                net.trainFcn='trainbfg';
                net.trainParam.epochs=1000;
            case 12
                net.trainFcn='traingda';
                net.trainParam.epochs=1000;
            case 13
                net.trainFcn='trainoss';
                net.trainParam.epochs=1000;
            case 14
                net.trainFcn='trainb';
                net.trainParam.epochs=1000;
            case 15
                net.trainFcn='traingdm';
                net.trainParam.epochs=1000;
            case 16
                net.trainFcn='trains';
                net.trainParam.epochs=1000;
            case 17
                net.trainFcn='traingd';
                net.trainParam.epochs=1000;
        end
%         net.trainFcn='trainbr';
%         net.divideFcn='';%Pruebas Mayo 2015
        net.divideFcn='divideind';
        if i==1
            net.divideParam.trainInd=[index(2):(index(end-nVal)-1)];
%             net.divideParam.valInd=[index(end-nVal):(index(end)-1)];
            net.divideParam.valInd=[index(1):(index(2)-1)];
            net.divideParam.testInd=[];
        elseif i==(m)
            net.divideParam.trainInd=[index(1):(index(i)-1)];
%             net.divideParam.valInd=[index(end-nVal):(index(end)-1)];
            net.divideParam.valInd=[index(i):(index(end)-1)];
            net.divideParam.testInd=[];
        else
            net.divideParam.trainInd=[index(1):(index(i)-1) index(i+1):(index(end-nVal)-1)];
%             net.divideParam.valInd=[index(end-nVal):(index(end)-1)];
            net.divideParam.valInd=[index(i):(index(i+1)-1)];
            net.divideParam.testInd=[];
        end
%         net.trainParam.epochs=1000;%Pruebas Mayo 2015
        net.inputs{1}.processFcns={'fixunknowns','removeconstantrows'};
        net.outputs{2}.processFcns={'removeconstantrows'};
        net.trainParam.showWindow=0;
%         net.layers{2}.transferFcn='tansig';%%Minor Revision 2016
       [net, tr] = train(net,inputs',targets');
%        hold off;
%        plotperf(tr);
%        ep=tr.num_epochs;
%     end
%     fprintf('Entrenamiento del modelo %d finalizado\n',i);
    
    %% Validación del modelo neuronal 
    nombreexp=strcat (folder,'/Experimento_',int2str(i),'.mat');
    ValE=load(nombreexp);
    NN=[];
    Xo=[];
    tsim=[];
    hits=0;
    hitsa=0;
    epsilon=0.02;
    [f,c]=size(ValE.experimento);

%     param=[0.024 0.361 17.87 2.138 1.919];
%     ac=[ValE.experimento(1:3,5)']; %HNN-RB8,9,10,12,13,14-->1:3, resto-->1:2
    ac=[];
    Tref=573;
    LW = net.LW{2,1};
    IW = net.IW{1};
    b1 = net.b{1};
    b2 = net.b{2};

    for j=1+d:f 
            switch mod
                case 1
                    inputNN=[ValE.experimento(j,1) ValE.experimento(j,2) ValE.experimento(j,3)];
                case 2
                    inputNN=[ValE.experimento(j-1,1) ValE.experimento(j,1) ValE.experimento(j-1,2) ValE.experimento(j,2) ValE.experimento(j-1,3) ValE.experimento(j,3)]; 
                case 3
                    if j==1+d
                        inputNN=[ValE.experimento(j-1,1) ValE.experimento(j,1) ValE.experimento(j-1,2) ValE.experimento(j,2) ValE.experimento(j-1,3) ValE.experimento(j,3) 0.2 0.2]; 
                    elseif j==2+d
                        inputNN=[ValE.experimento(j-1,1) ValE.experimento(j,1) ValE.experimento(j-1,2) ValE.experimento(j,2) ValE.experimento(j-1,3) ValE.experimento(j,3) 0.2 NN(j-1-d)];
                    else
                        inputNN=[ValE.experimento(j-1,1) ValE.experimento(j,1) ValE.experimento(j-1,2) ValE.experimento(j,2) ValE.experimento(j-1,3) ValE.experimento(j,3) NN(j-2-d) NN(j-1-d)];
                    end
                case 4
                    if j==1+d
                        inputNN=[ValE.experimento(j-1,1) ValE.experimento(j,1) ValE.experimento(j-1,2) ValE.experimento(j,2) ValE.experimento(j-1,3) ValE.experimento(j,3) ValE.experimento(j-1,4) ValE.experimento(j,4) 0.2 0.2]; 
                    elseif j==2+d
                        inputNN=[ValE.experimento(j-1,1) ValE.experimento(j,1) ValE.experimento(j-1,2) ValE.experimento(j,2) ValE.experimento(j-1,3) ValE.experimento(j,3) ValE.experimento(j-1,4) ValE.experimento(j,4) 0.2 NN(j-1-d)];
                    else
                        inputNN=[ValE.experimento(j-1,1) ValE.experimento(j,1) ValE.experimento(j-1,2) ValE.experimento(j,2) ValE.experimento(j-1,3) ValE.experimento(j,3) ValE.experimento(j-1,4) ValE.experimento(j,4) NN(j-2-d) NN(j-1-d)];
                    end
                case 5
                    if j==1
                        inputNN=[ValE.experimento(j,1) ValE.experimento(j,2) ValE.experimento(j,3) 0.2];
                    else
                        inputNN=[ValE.experimento(j,1) ValE.experimento(j,2) ValE.experimento(j,3) NN(j-1)];
                    end
                case 6
                    if j==1
                        inputNN=[ValE.experimento(j,1) ValE.experimento(j,2) ValE.experimento(j,3) 0.2 0.2];
                    elseif j==2
                        inputNN=[ValE.experimento(j,1) ValE.experimento(j,2) ValE.experimento(j,3) 0.2 NN(j-1)];
                    else
                        inputNN=[ValE.experimento(j,1) ValE.experimento(j,2) ValE.experimento(j,3) NN(j-2) NN(j-1)];
                    end
                case 7
                    if j==1
                        inputNN=[ValE.experimento(j,1) ValE.experimento(j,2) ValE.experimento(j,3) 0.2 0.2 0.2];
                    elseif j==2
                        inputNN=[ValE.experimento(j,1) ValE.experimento(j,2) ValE.experimento(j,3) 0.2 0.2 NN(j-1)];
                    elseif j==3
                        inputNN=[ValE.experimento(j,1) ValE.experimento(j,2) ValE.experimento(j,3) 0.2 NN(j-2) NN(j-1)];
                    else
                        inputNN=[ValE.experimento(j,1) ValE.experimento(j,2) ValE.experimento(j,3) NN(j-3) NN(j-2) NN(j-1)];
                    end
            end
            
            NN=[NN purelin(LW*tansig((IW*inputNN')+b1)+b2)];
%             NN=[NN tansig(LW*tansig((IW*inputNN')+b1)+b2)];%%Minor Revision 2016
            
%             NN=[NN sim(net,inputNN')];
            %HNN-RB1,2,3,5,6,11
    %         Xo(j-2)=ValE.experimento(j,6);
    %         Error_Xo(j-2)=Xo(j-2)-NN(j-2);
    %         if NN(j-2)>(Xo(j-2)-epsilon) && NN(j-2)<(Xo(j-2)+epsilon)
    %             hits=hits+1;
    %         end
            %HNN-RB4
    %         Xo(j-1)=ValE.experimento(j,6);
    %         Error_Xo(j-1)=Xo(j-1)-NN(j-1);
    %         if NN(j-1)>(Xo(j-1)-epsilon) && NN(j-1)<(Xo(j-1)+epsilon)
    %             hits=hits+1;
    %         end
            %HNN-RB7,8,9,10,12,13,14, 01-2015
%             Xo(j-3)=ValE.experimento(j,6);
%             Error_Xo(j-3)=Xo(j-3)-NN(j-3);
%             if NN(j-3)>(Xo(j-3)-epsilon) && NN(j-3)<(Xo(j-3)+epsilon)
%                 hits=hits+1;
%             end
            %HNN-RB02-2015
            Xo =[Xo ValE.experimento(j,6)];
            tsim = [tsim mapminmax('reverse',ValE.experimento(j,4),Xn4ps)];
            if NN(j-d)>(Xo(j-d)-epsilon) && NN(j-d)<(Xo(j-d)+epsilon)
                hits=hits+1;
            end
%         end
    end
    %Errores y Representación
    Error_Xo = Xo - NN;
    MAE(i)=mean(abs(Error_Xo));
    RMSE(i)=sqrt(mean(Error_Xo.*Error_Xo));
    MAX(i)=max(abs(Error_Xo));
    MAPE(i)=mean(abs(Error_Xo./Xo))*100.0;
    SSE(i)=sum(Error_Xo.^2);
    %MAPE(i)=mean(abs(Error_Xo./mean(Xo)))*100.0;%Formula MAPE alternativa.
    HITS(i)=(hits*100)/length(Error_Xo);
%     fprintf('\nErrores Xo - Curva Validación: %d',i);
%     fprintf('\n**Redes neuronales**');
% %    fprintf('\nEpochs: %d',tr.num_epochs);
%     fprintf('\nRMSE: %f',RMSE(i));
%     fprintf('\nMAE: %f',MAE(i));
%     fprintf('\nMAX: %f',MAX(i));
%     fprintf('\nMAPE: %f',MAPE(i));
%     fprintf('\nSSE: %f',SSE(i));
%     fprintf('\nHITS: %f\n',HITS(i));
%     

%     h=figure(i);
%     hold on;
%     plot(tsim,Xo,'ok');
%     plot(tsim,NN,'-.b');
%     legend('Xo','NN(Xo)');
%     xlim([0 tsim(end)]);
%     ylim([0 1]);
%     textofig = sprintf('Ti = %f\nTf = %f\nXw = %f\ntao = %f',((ValE.experimento(1,2)-Xn2ps.ymin)*(Xn2ps.xmax-Xn2ps.xmin)/(Xn2ps.ymax-Xn2ps.ymin))+Xn2ps.xmin,((ValE.experimento(end,2)-Xn2ps.ymin)*(Xn2ps.xmax-Xn2ps.xmin)/(Xn2ps.ymax-Xn2ps.ymin))+Xn2ps.xmin,((ValE.experimento(1,1)-Xn1ps.ymin)*(Xn1ps.xmax-Xn1ps.xmin)/(Xn1ps.ymax-Xn1ps.ymin))+Xn1ps.xmin,((ValE.experimento(1,3)-Xn3ps.ymin)*(Xn3ps.xmax-Xn3ps.xmin)/(Xn3ps.ymax-Xn3ps.ymin))+Xn3ps.xmin);
%     text(2,0.4,textofig);
%     titulo=strcat('Xo E',int2str(i));
%     title(titulo);
%      
%     estim=strcat ('Results\NN_',int2str(i),'_',int2str(neuronas),'_',int2str(iter),'.mat');
%     save(estim,'NN');
%     figur=strcat('Results\FigE',int2str(i),'_',int2str(neuronas),'_',int2str(iter),'.fig');
%     saveas(figure(i),figur);
%     close(h);
    clear NN;
    clear Xo;
    clear tsim;
    clear a;
    clear ac;
    clear Xo;
    clear Error_Xo;
    clear tsim;
    
end
% view(net);
% valAIC=aic(net);
%% Presentación de resultados: consola, txt y Excel
% fprintf('\n\nErrores medios Xo - %d exp. - mod: %d - iter: %d - neuronas: %d',m,mod,iter,neuronas);
% fprintf('\n**Redes neuronales**');
% fprintf('\nRMSE: %f',mean(RMSE));
% fprintf('\nMAE: %f',mean(MAE));
% fprintf('\nMAX: %f',max(MAX));
% fprintf('\nMAPE: %f',mean(MAPE));
% fprintf('\nSSE: %f\n',sum(SSE));
% fprintf('\nHITS: %f\n',mean(HITS));
% 
% fprintf('\n\nMAPE Tcte: %f',mean(MAPE(1:14)));
% fprintf('\nMAPE Trampa: %f',mean(MAPE(15:23)));

exceldata={'Experimento Xo','RMSE','MAE','MAX','MAPE', 'SSE','HITS'};
% excelfile=strcat(pwd,'/',folder,'/ANNRealDataLOOCV','Estrc_',int2str(neuronas),'.xlsx');%neuronas
excelfile=strcat(pwd,'/',folder,'/ANNRealDataLOOCV','Estrc_',int2str(trfcn),'.xlsx');%trainfcn
% xlswrite(excelfile,exceldata,iter,'A1');
% xlswrite(excelfile,[[1:m]' RMSE' MAE' MAX' MAPE' SSE' HITS'],iter,'A2');
% xlsrange=strcat('A',num2str(m+2));
% xlswrite(excelfile,{'Total',mean(RMSE),mean(MAE),max(MAX),mean(MAPE),sum(SSE),mean(HITS)},iter,xlsrange);
csvwrite(excelfile,exceldata);
csvwrite(excelfile,[[1:m]' RMSE' MAE' MAX' MAPE' SSE' HITS'],1,0);
csvwrite(excelfile,{mean(RMSE),mean(MAE),max(MAX),mean(MAPE),sum(SSE),mean(HITS)},m+2,1);
MAPEF = mean(MAPE);
RMSEF = mean(RMSE);

% fprintf('\n\n Ejecución finalizada correctamente');
% fprintf('\n Resultados guardados en: %s \n',excelfile);
