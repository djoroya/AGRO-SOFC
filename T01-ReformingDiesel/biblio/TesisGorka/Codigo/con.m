harm = [1;3];
tspan =linspace(0,pi/2,200);
G = @(p,t,eps) eps-sum(p.*sin(harm.*t));

theta = linspace(0,2*pi,100);

eps = 1e-1;


figure(1)
clf
subplot(1,3,1)
hold on 
plot1 = plot(tspan,G(p0,tspan,eps));
title('G(p,\tau)')
ylim([-10*eps 10*eps])
yline(0)
%
subplot(1,3,2)
plot2  = plot(tspan,-sign(G(p0,tspan,eps)));
title('f(\tau)')
ylim([-1.5 1.5])
daspect([1 1 1])

hold on 
subplot(1,3,3)
daspect([1 1 1])

hold on 
plot(0,0,'r','marker','.','markersize',18)
xline(0)
yline(0)




for irr = linspace(eps,100*eps,5) 
for itheta = theta
p0 = irr*[sin(itheta);cos(itheta)];

plot1.YData = G(p0,tspan,eps);

plot2.YData = -sign(G(p0,tspan,eps));

subplot(1,3,3)
f_fcn = @(t) interp1(tspan,-sign(G(p0,tspan,eps)),t);
[~,beta] = ode23(@(t,beta) (4/pi)*sin(harm*t)*f_fcn(t),tspan,p0);
beta =flipud(beta);
plot(beta(:,1),beta(:,2))
plot(beta(1,1),beta(1,2),'color','k','marker','.','markerSize',18)

title('traj')
pause(0.01)
end
end