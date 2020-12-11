%% Import data from text file
% Script for importing data from the following text file:
%
%    filename: /Users/djoroya/Dropbox/My Mac (Deyviss’s MacBook Pro)/Documents/GitHub/AGRO-SOFC/code/Model&Optimization/data/19_Data_Ainhoa_SOFC_SingleCell_ESC/SPS_Data/2020_10_12_075.txt
%
% Auto-generated by MATLAB on 10-Dec-2020 10:08:18


function DataSOFCtable = DataSOFC(filename)
opts = delimitedTextImportOptions("NumVariables", 67);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = "\t";

% Specify column names and types
opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9", "VarName10", "VarName11", "VarName12", "VarName13", "VarName14", "VarName15", "VarName16", "VarName17", "VarName18", "VarName19", "VarName20", "VarName21", "VarName22", "VarName23", "VarName24", "VarName25", "VarName26", "VarName27", "VarName28", "VarName29", "VarName30", "VarName31", "VarName32", "VarName33", "VarName34", "VarName35", "VarName36", "VarName37", "VarName38", "VarName39", "VarName40", "VarName41", "VarName42", "VarName43", "VarName44", "VarName45", "VarName46", "VarName47", "VarName48", "VarName49", "VarName50", "VarName51", "VarName52", "VarName53", "VarName54", "VarName55", "VarName56", "VarName57", "VarName58", "VarName59", "VarName60", "VarName61", "VarName62", "VarName63", "VarName64", "VarName65", "ENDSAMPLEON", "QQ-6"];

opts.VariableNames = ["Date" , ...
"Time" , ...
"T_Oven-01" , ...
"T_Oven-02" , ...
"T_A-01" , ...
"T_A-02" , ...
"T_A-03" , ...
"T_A-04" , ...
"T_A-05" , ...
"T_A-06" , ...
"T_C-01" , ...
"T_C-02" , ...
"T_C-03" , ...
"T_C-04" , ...
"T_C-05" , ...
"T_C-06" , ...
"T_A_In" , ...
"T_A_Out" , ...
"T_C_In" , ...
"T_C_Out" , ...
"T Humidifier" , ...
"Vol H2O (actual)", ...
"p Hum (actual)" , ...
"i_target" , ...
"i_act" , ...
"i_act2" , ...
"A" , ...
"V_act" , ...
"N2_tar" , ...
"N2_act" , ...
"H2_tar" , ...
"H2_act" , ...
"CO_tar" , ...
"CO_act" , ...
"CO2_tar" , ...
"CO2_act" , ...
"CH4_tar" , ...
"CH4_act" , ...
"RG_tar" , ...
"RG_act" , ...
"QQ-1" , ...
"QQ-2" , ...
"Air_tar" , ...
"Air_act" , ...
"O2_tar" , ...
"O2_act" , ...
"N2_Air_tar", ...
"N2_Air_act" , ...
"QQ-3" , ...
"v_H2_act" , ...
"v_CO_act" , ...
"v_CO2_act" , ...
"v_CH4_act" , ...
"v_O2_act" , ...
"QQ-4" , ...
"p_atm" , ...
"v_O2_cath" , ...
"stat-GAS-SUPPLY" , ...
"stat-SOFC-OVEN" , ...
"ALARM-fuelflow" , ...
"ALARM-pmax" , ...
"ALARM-dtover1" , ...
"ALARM-dtover1" , ...
"ALARM-load" , ...
"ALARM-modbus" , ...
"QQ-5" , ...
"QQ-6" ];

opts.VariableTypes = ["datetime", "datetime", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "categorical", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "QQ-6", "WhitespaceRule", "preserve");
%opts = setvaropts(opts, ["ENDSAMPLEON", "QQ-6"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, "Date", "InputFormat", "dd.MM.yyyy");
opts = setvaropts(opts, "Time", "InputFormat", "HH:mm:ss");

% Import the data
DataSOFCtable= readtable(path +"/data/19_Data_Ainhoa_SOFC_SingleCell_ESC/SPS_Data/"+filename, opts);
DataSOFCtable= DataSOFCtable(~isnat(DataSOFCtable.Date),:);

%% Clear temporary variables
end
%%
