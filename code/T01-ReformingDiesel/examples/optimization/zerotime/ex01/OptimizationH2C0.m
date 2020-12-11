clear
load('/Users/djoroya/Dropbox/My Mac (Deyviss’s MacBook Pro)/Documents/GitHub/AGRO-SOFC/code/Model&Optimization/data/models/ANN-ZeroTime.mat')

%%
%    Inputs
%    {'Temperature'}    {'RatioSteamC'}    {'RatioOC'}    {'RatioHC'}
%     Outputs
%     {'CO'}    {'H2'}

import casadi.*

Ts   = MX.sym('T' );
SCs  = MX.sym('SC');
OCs  = MX.sym('OC');
HCs  = MX.sym('HC');

u = [Ts;SCs;OCs;HCs];
