%%
clear 

k= [1, 0.5]';
coeff = 1.4064;

c = (2:-1:1)' + rand(2,1);

nquarter = 4;
pi = 500*rand(2,nquarter,12);

sq = @(p) sqrt(sum( (p < pi).*(pi - p).^2,2));

phi = @(p) sum(p.*c + coeff*k.*sq(p),1);
Cost = @(p) sum(phi(p),3);

%%
p0 = floor(500*rand(2,1));
p0 = sort(p0);
p0 = reshape(p0,2,1,1);
Cost(p0)
%%
%
Np1 = 500;
Np2 = 500;
%
p1_span = linspace(0,1000,Np1);
p2_span = linspace(0,1000,Np2);

C = zeros(Np1,Np2);
for ip1 = 1:Np1
    for ip2 = 1:Np2
        p0 = [p1_span(ip1);p2_span(ip2)];
        C(ip1,ip2) = Cost(p0);
    end    
end
%%
%close all
clf
hold on 
xline(450,'LineWidth',4,'Color','w')
line(p1_span,p1_span,1e5 + 0*p1_span,'LineWidth',4,'Color','w')
surf(p1_span,p2_span,C')
zlim([0 1e5])
caxis([0 5e4])
xlabel('p_1')
ylabel('p_2')
zlabel('Cost')
shading interp
view(0,90)
lightangle(190,20)
lightangle(20,20)
colorbar

%%
[p1_ms,p2_ms] = meshgrid(p1_span,p2_span);
clf
hold on

xline(450,'LineWidth',4,'Color','w')
plot(p1_span,p1_span,'LineWidth',4,'Color','w')
surf(p1_ms,p2_ms,(450<p1_ms).*(p1_ms<p2_ms))

colorbar 
view(90,-90)
xlabel('p_1')
ylabel('p_2')
colormap jet
shading interp

%%
import casadi.*

psym = SX.sym('p',[2 1]);


g = diff(psym);
g = [psym(1) - 450;g]