function [testRMSE,testMAPE]=ANN_FF_RealData(mod,neuronas,trfcn,cont)

% clear all;
% close all;
% clc;

%% Recogida de los datos del proceso BTO
% datos=xlsread('Datasets/ExpData_BTOAT10.xlsx',2,'F2:K389');%Exp: 1-23

% datos=xlsread('Datasets/ExpData_BTOAT10_2.xlsx',2,'F2:K518');%Exp: 1-23 + 2 Val
datos=xlsread('Datasets/ExpData_BTOAT10_4.xlsx',2,'F2:K389');%Exp: 1-21 + 2 Val
nVal=0;
datosTest=xlsread('Datasets/ExpData_BTOAT10_2.xlsx',2,'F519:K628');%Exp: Test 1-3

% datos=xlsread('Datasets/ExpData_BTOAT10_3.xlsx',2,'F2:K587');%Exp: 1-21 + 2 Val
% nVal=2;
% datosTest=xlsread('Datasets/ExpData_BTOAT10_3.xlsx',2,'F588:K622');%Exp: Test 1-2

% datosVal=xlsread('Datasets/ExpData_BTOAT10.xlsx',2,'F390:K622');%Exp: Val 1-2
% datos=xlsread('Datasets\Datos_BTOmod1tiempo-SIMULACION.xlsx',1,'A2:F1153');
% datos=xlsread('Datasets\Datos_BTOmod1tiempo-SIMULACION.xlsx',1,'A2:F1264');%Exp: 1-42 
save('Results/datos_BTO.mat','datos');
% fprintf('Datos cargados correctamente\n');
%% Normalización de los datos
maxdat1=max(datos(:,1));
maxdat2=max(datos(:,2));
maxdat3=max(datos(:,3));
maxdat4=max(datos(:,4));
% x1_again = mapminmax('reverse',y1,PS)
[datos(:,1),Xn1ps]=mapminmax(datos(:,1)');
[datos(:,2),Xn2ps]=mapminmax(datos(:,2)');
[datos(:,3),Xn3ps]=mapminmax(datos(:,3)');
[datos(:,4),Xn4ps]=mapminmax(datos(:,4)');
% datos(:,1)=datos(:,1)/max(datos(:,1));
% datos(:,2)=datos(:,2)/max(datos(:,2));
% datos(:,3)=datos(:,3)/max(datos(:,3));
% datos(:,4)=datos(:,4)/max(datos(:,4));
save('Results/datos_BTOnorm.mat','datos');

datosTest(:,1)=mapminmax('apply',datosTest(:,1)',Xn1ps);
datosTest(:,2)=mapminmax('apply',datosTest(:,2)',Xn2ps);
datosTest(:,3)=mapminmax('apply',datosTest(:,3)',Xn3ps);
datosTest(:,4)=mapminmax('apply',datosTest(:,4)',Xn4ps);
% datosVal(:,1)=mapminmax('apply',datosVal(:,1)',Xn1ps);
% datosVal(:,2)=mapminmax('apply',datosVal(:,2)',Xn2ps);
% datosVal(:,3)=mapminmax('apply',datosVal(:,3)',Xn3ps);
% datosVal(:,4)=mapminmax('apply',datosVal(:,4)',Xn4ps);
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
    nombreexp=strcat ('Results/Experimento_',int2str(i),'.mat');
    save(nombreexp,'experimento');
    switch mod
        case 1 %ANN 1
            inputs=[inputs ; experimento(:,1) experimento(:,2) experimento(:,3) ];
            targets=[targets ; experimento(:,6)];
            if i==1
                index=[index (1+length(experimento))];
            else
                index=[index (index(i)+length(experimento))];
            end
            d=0;
        case 2 %ANN 2
            inputs=[inputs ; experimento(1:end-1,1) experimento(2:end,1) experimento(1:end-1,2) experimento(2:end,2) experimento(1:end-1,3) experimento(2:end,3) ];
            targets=[targets ; experimento(2:end,6)];
            if i==1
                index=[index (1+length(experimento)-1)];
            else
                index=[index (index(i)+length(experimento)-1)];
            end
            d=1;
        case 3 %ANN 3
            inputs=[inputs ; experimento(2:end-1,1) experimento(3:end,1) experimento(2:end-1,2) experimento(3:end,2) experimento(2:end-1,3) experimento(3:end,3) experimento(1:end-2,6) experimento(2:end-1,6)];
            targets=[targets ; experimento(3:end,6)];
            if i==1
                index=[index (1+length(experimento)-2)];
            else
                index=[index (index(i)+length(experimento)-2)];
            end
            d=1;
        case 4 %ANN 4
            inputs=[inputs ; experimento(2:end-1,1) experimento(3:end,1) experimento(2:end-1,2) experimento(3:end,2) experimento(2:end-1,3) experimento(3:end,3) experimento(2:end-1,4) experimento(3:end,4) experimento(1:end-2,6) experimento(2:end-1,6)];
            targets=[targets ; experimento(3:end,6)];
            if i==1
                index=[index (1+length(experimento)-2)];
            else
                index=[index (index(i)+length(experimento)-2)];
            end
            d=1;
    end
end
nombreinput=strcat ('Results/Input.mat');
nombretarget=strcat ('Results/Target.mat');
save(nombreinput,'inputs');
save(nombretarget,'targets');
save('Results/ExpIndex.mat','index');

indicest=find(datosTest(:,5)>=1);
[mt,nt]=size(indicest);
for i=1:mt
    if i~=mt
        experimentoTest=datosTest(indicest(i):(indicest(i+1)-1),:);    
    else
        experimentoTest=datosTest(indicest(i):end,:);
    end    
    nombreexp=strcat ('Results/ExperimentoTest_',int2str(i),'.mat');
    save(nombreexp,'experimentoTest');
end
fprintf('\n+-->Train Experiments: %d - Val. Experiments: %d - Test Experiments: %d\n',m-nVal,nVal,mt);
% fprintf('Datos preparados correctamente\n+-->Se han encontrado %d experimentos\n',m);
%%
% filename=strcat(pwd,'\ResultadosPruebas\ResultadosRNAhibridoRealDataLOOCV.txt');
% fileID=fopen(filename,'w','n','UTF-8');

    %% Entrenamiento del modelo neuronal
        net = newff(inputs',targets',neuronas);%11
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
        net.divideFcn='divideind';

            net.divideParam.trainInd=[index(1):(index(end-nVal)-1)];  
%             net.divideParam.valInd=[index(end-nVal):(index(end)-1)];
%             net.divideParam.valInd=[index(1):(index(2)-1)];
            net.divideParam.valInd=[];
            net.divideParam.testInd=[];

%         net.trainParam.epochs=1000;%Pruebas Mayo 2015
        net.inputs{1}.processFcns={'fixunknowns','removeconstantrows'};
        net.outputs{2}.processFcns={'removeconstantrows'};
        net.layers{2}.transferFcn='logsig';%%Minor Revision 2016
        %net.trainParam.showWindow=0;
       [net, tr] = train(net,inputs',targets');
%        hold off;
%        plotperf(tr);
%        ep=tr.num_epochs;
%     end
%     fprintf('Entrenamiento del modelo %d finalizado\n',i);
    normVal=[Xn1ps, Xn2ps, Xn3ps, Xn4ps];
    save('Results/BTOnet.mat','normVal','net');
    %% Validación del modelo neuronal
    %Error de entrenamiento
    TrainXototal=[];
    TrainNNtotal=[];
    for i=1:m-nVal
        nombreexp=strcat ('Results/Experimento_',int2str(i),'.mat');
        ValE=load(nombreexp);
        NN=[];
        Xo=[];
        tsim=[];
        hits=0;
        epsilon=0.02;
        [f,c]=size(ValE.experimento);
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
            end

%                 NN=[NN purelin(LW*tansig((IW*inputNN')+b1)+b2)];
                NN=[NN logsig(LW*tansig((IW*inputNN')+b1)+b2)];%%Minor Revision 2016

                %ANN
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
%         fprintf('\nErrores Xo - Curva Validación: %d',i);
    %     fprintf('\n**Redes neuronales**');
    % %    fprintf('\nEpochs: %d',tr.num_epochs);
        fprintf('\nError de Entrenamiento - Exp. %d',i);
        fprintf('\nRMSE: %f',RMSE(i));
    %     fprintf('\nMAE: %f',MAE(i));
    %     fprintf('\nMAX: %f',MAX(i));
        fprintf('\nMAPE: %f',MAPE(i));
    %     fprintf('\nSSE: %f',SSE(i));
    %     fprintf('\nHITS: %f\n',HITS(i));
    %     
        h= figure(i);
        hold on;
        plot(tsim,Xo,'ok');
        plot(tsim,NN,'-.b');
        legend('Xo','NN(Xo)');
        xlim([0 tsim(end)]);
        ylim([0 1]);
        textofig = sprintf('Ti = %f\nTf = %f\nXw = %f\ntao = %f',((ValE.experimento(1,2)-Xn2ps.ymin)*(Xn2ps.xmax-Xn2ps.xmin)/(Xn2ps.ymax-Xn2ps.ymin))+Xn2ps.xmin,((ValE.experimento(end,2)-Xn2ps.ymin)*(Xn2ps.xmax-Xn2ps.xmin)/(Xn2ps.ymax-Xn2ps.ymin))+Xn2ps.xmin,((ValE.experimento(1,1)-Xn1ps.ymin)*(Xn1ps.xmax-Xn1ps.xmin)/(Xn1ps.ymax-Xn1ps.ymin))+Xn1ps.xmin,((ValE.experimento(1,3)-Xn3ps.ymin)*(Xn3ps.xmax-Xn3ps.xmin)/(Xn3ps.ymax-Xn3ps.ymin))+Xn3ps.xmin);
        text(2,0.4,textofig);
        titulo=strcat('Xo E',int2str(i));
        title(titulo);
    %         
        estim=strcat ('Results/NN_',int2str(i),'_cont',int2str(cont),'.mat');
        save(estim,'NN');
        figur=strcat('Results/FigE',int2str(i),'_cont',int2str(cont),'.fig');
        saveas(figure(i),figur);
        close (h);
        TrainXototal = [TrainXototal; Xo'];
        TrainNNtotal = [TrainNNtotal; NN'];
        clear NN;
        clear ac;
        clear Xo;
        clear Error_Xo;
        clear tsim;

    end
    % Presentación de resultados: consola, txt y Excel
    % fprintf('\n\nErrores medios Xo - Totales: %d experimentos',m);
    % fprintf('\n**Redes neuronales**');
    fprintf('\n***Error de Entrenamiento: %d exp.',m-nVal);
    fprintf('\nRMSE: %.4f',mean(RMSE));
    % fprintf('\nMAE: %f',mean(MAE));
    % fprintf('\nMAX: %f',max(MAX));
    fprintf('\nMAPE: %.4f',mean(MAPE));
    % fprintf('\nSSE: %f\n',sum(SSE));
    % fprintf('\nHITS: %f\n',mean(HITS));
    % 
    % fprintf('\n\nMAPE Tcte: %f',mean(MAPE(1:14)));
    % fprintf('\nMAPE Trampa: %f',mean(MAPE(15:23)));

    exceldata={'Experimento Xo','RMSE','MAE','MAX','MAPE', 'SSE','HITS'};
    excelfile=strcat(pwd,'/Results/ANNRealData','_cont',int2str(cont),'.xlsx');
    xlswrite(excelfile,exceldata,'Train','A1');
    xlswrite(excelfile,[[1:(m-nVal)]' RMSE' MAE' MAX' MAPE' SSE' HITS'],'Train','A2');

    clear RMSE;
    clear MAE;
    clear MAX;
    clear MAPE;
    clear SSE;
    clear HITS;
    h=figure(200);
    plotregression(TrainXototal,TrainNNtotal,'Train ANN');
    saveas(h,strcat('Results/CurvRegrTrain','_cont',int2str(cont),'.fig'));
    close(h);
    
%Validación TEST
fprintf('\n\n*** VALIDATION ***');
TestXototal=[];
TestNNtotal=[];
for i=1:mt
    nombreexp=strcat ('Results/ExperimentoTest_',int2str(i),'.mat');
    ValE=load(nombreexp);
        NN=[];
        Xo=[];
        tsim=[];
        hits=0;
        epsilon=0.02;
        [f,c]=size(ValE.experimentoTest);
        LW = net.LW{2,1};
        IW = net.IW{1};
        b1 = net.b{1};
        b2 = net.b{2};

        for j=1+d:f 
            switch mod
                case 1
                    inputNN=[ValE.experimentoTest(j,1) ValE.experimentoTest(j,2) ValE.experimentoTest(j,3)];
                case 2
                    inputNN=[ValE.experimentoTest(j-1,1) ValE.experimentoTest(j,1) ValE.experimentoTest(j-1,2) ValE.experimentoTest(j,2) ValE.experimentoTest(j-1,3) ValE.experimentoTest(j,3)]; 
                case 3
                    if j==1+d
                        inputNN=[ValE.experimentoTest(j-1,1) ValE.experimentoTest(j,1) ValE.experimentoTest(j-1,2) ValE.experimentoTest(j,2) ValE.experimentoTest(j-1,3) ValE.experimentoTest(j,3) 0.2 0.2]; 
                    elseif j==2+d
                        inputNN=[ValE.experimentoTest(j-1,1) ValE.experimentoTest(j,1) ValE.experimentoTest(j-1,2) ValE.experimentoTest(j,2) ValE.experimentoTest(j-1,3) ValE.experimentoTest(j,3) 0.2 NN(j-1-d)];
                    else
                        inputNN=[ValE.experimentoTest(j-1,1) ValE.experimentoTest(j,1) ValE.experimentoTest(j-1,2) ValE.experimentoTest(j,2) ValE.experimentoTest(j-1,3) ValE.experimentoTest(j,3) NN(j-2-d) NN(j-1-d)];
                    end
                case 4
                    if j==1+d
                        inputNN=[ValE.experimentoTest(j-1,1) ValE.experimentoTest(j,1) ValE.experimentoTest(j-1,2) ValE.experimentoTest(j,2) ValE.experimentoTest(j-1,3) ValE.experimentoTest(j,3) ValE.experimentoTest(j-1,4) ValE.experimentoTest(j,4) 0.2 0.2]; 
                    elseif j==2+d
                        inputNN=[ValE.experimentoTest(j-1,1) ValE.experimentoTest(j,1) ValE.experimentoTest(j-1,2) ValE.experimentoTest(j,2) ValE.experimentoTest(j-1,3) ValE.experimentoTest(j,3) ValE.experimentoTest(j-1,4) ValE.experimentoTest(j,4) 0.2 NN(j-1-d)];
                    else
                        inputNN=[ValE.experimentoTest(j-1,1) ValE.experimentoTest(j,1) ValE.experimentoTest(j-1,2) ValE.experimentoTest(j,2) ValE.experimentoTest(j-1,3) ValE.experimentoTest(j,3) ValE.experimentoTest(j-1,4) ValE.experimentoTest(j,4) NN(j-2-d) NN(j-1-d)];
                    end
            end

%                 NN=[NN purelin(LW*tansig((IW*inputNN')+b1)+b2)];
                NN=[NN logsig(LW*tansig((IW*inputNN')+b1)+b2)];%%Minor Revision 2016

                %ANN
                Xo =[Xo ValE.experimentoTest(j,6)];
                tsim = [tsim mapminmax('reverse',ValE.experimentoTest(j,4),Xn4ps)];
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
%         fprintf('\nErrores Xo - Curva Validación: %d',i);
    %     fprintf('\n**Redes neuronales**');
    % %    fprintf('\nEpochs: %d',tr.num_epochs);
        fprintf('\nError de Test - Exp. %d',i);
        fprintf('\nRMSE: %f',RMSE(i));
    %     fprintf('\nMAE: %f',MAE(i));
    %     fprintf('\nMAX: %f',MAX(i));
        fprintf('\nMAPE: %f',MAPE(i));
    %     fprintf('\nSSE: %f',SSE(i));
    %     fprintf('\nHITS: %f\n',HITS(i));
    %     
        h= figure(i+100);
        hold on;
        plot(tsim,Xo,'ok');
        plot(tsim,NN,'-.b');
        legend('Xo','NN(Xo)');
        xlim([0 tsim(end)]);
        ylim([0 1]);
        textofig = sprintf('Ti = %f\nTf = %f\nXw = %f\ntao = %f',((ValE.experimentoTest(1,2)-Xn2ps.ymin)*(Xn2ps.xmax-Xn2ps.xmin)/(Xn2ps.ymax-Xn2ps.ymin))+Xn2ps.xmin,((ValE.experimentoTest(end,2)-Xn2ps.ymin)*(Xn2ps.xmax-Xn2ps.xmin)/(Xn2ps.ymax-Xn2ps.ymin))+Xn2ps.xmin,((ValE.experimentoTest(1,1)-Xn1ps.ymin)*(Xn1ps.xmax-Xn1ps.xmin)/(Xn1ps.ymax-Xn1ps.ymin))+Xn1ps.xmin,((ValE.experimentoTest(1,3)-Xn3ps.ymin)*(Xn3ps.xmax-Xn3ps.xmin)/(Xn3ps.ymax-Xn3ps.ymin))+Xn3ps.xmin);
        text(2,0.4,textofig);
        titulo=strcat('Xo E',int2str(i));
        title(titulo);
    %         
        estim=strcat ('Results/TestNN_',int2str(i),'_cont',int2str(cont),'.mat');
        save(estim,'NN');
        figur=strcat('Results/FigTestE',int2str(i),'_cont',int2str(cont),'.fig');
        saveas(h,figur);
        close (h);
        TestXototal=[TestXototal; Xo'];
        TestNNtotal=[TestNNtotal; NN'];
        clear NN;
        clear ac;
        clear Xo;
        clear Error_Xo;
        clear tsim;
end

    fprintf('\n***Error de Test: %d exp.',mt);
    fprintf('\nRMSE: %.4f',mean(RMSE));
    % fprintf('\nMAE: %f',mean(MAE));
    % fprintf('\nMAX: %f',max(MAX));
    fprintf('\nMAPE: %.4f\n',mean(MAPE));
    % fprintf('\nSSE: %f\n',sum(SSE));
    % fprintf('\nHITS: %f\n',mean(HITS));
    % 
    % fprintf('\n\nMAPE Tcte: %f',mean(MAPE(1:14)));
    % fprintf('\nMAPE Trampa: %f',mean(MAPE(15:23)));

    exceldata={'Experimento Xo','RMSE','MAE','MAX','MAPE', 'SSE','HITS'};
    excelfile=strcat(pwd,'/Results/ANNRealData','_cont',int2str(cont),'.xlsx');
    xlswrite(excelfile,exceldata,'Test','A1');
    xlswrite(excelfile,[[1:mt]' RMSE' MAE' MAX' MAPE' SSE' HITS'],'Test','A2');

    h=figure(300);
    plotregression(TestXototal,TestNNtotal,'Test ANN');
    saveas(h,strcat('Results/CurvRegrTest','_cont',int2str(cont),'.fig'));
    close(h);
    
    h=figure(400);
    XoTotal = [TrainXototal; TestXototal];
    NNTotal = [TrainNNtotal; TestNNtotal];
%     plotregression(TrainXototal,TrainNNtotal,'Train',TestXototal,TestNNtotal,'Test');
    plotregression(XoTotal,NNTotal);
    saveas(h,strcat('Results/CurvRegr','_cont',int2str(cont),'.fig'));
    close(h);
    
    testMAPE = mean(MAPE);
    testRMSE = mean(RMSE);
end