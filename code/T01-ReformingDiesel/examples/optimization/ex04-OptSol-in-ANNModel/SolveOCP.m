function [ut_opt,xt_opt,xt_free] = SolveOCP(A,B,C,tspan,x0)
 %%
Nt = length(tspan);

[N,~] = size(A);
[~,M] = size(B);


import casadi.*
us = SX.sym('u',M,Nt-1);
x0s = SX.sym('x0',N,1); 
xnext = x0s;

for it = 2:Nt
   dt = tspan(it) - tspan(it-1);
   xnext = xnext + dt*(A*xnext + B*us(:,it-1) + C);
end
F = Function('F',{x0s,us},{xnext});
%
nlp = struct('x',us(:), 'f',1e-9*sum(sum(us.^2))+norm(F(x0,us))^2, 'g',[]);
S = nlpsol('S', 'ipopt', nlp);
%%
u0 = zeros(M,Nt-1);
sol = S('x0',u0(:));
ut_opt = full(reshape(sol.x,M,Nt-1));
%%
xt_opt = zeros(N,Nt);
xt_opt(:,1) = x0;
for it = 2:Nt
   dt = tspan(it) - tspan(it-1);
   xt_opt(:,it) = xt_opt(:,it-1) + dt*(A*xt_opt(:,it-1) + B*ut_opt(:,it-1) + C);
end
%%
xt_free = zeros(N,Nt);
xt_free(:,1) = x0;
for it = 2:Nt
   dt = tspan(it) - tspan(it-1);
   xt_free(:,it) = xt_free(:,it-1) + dt*(A*xt_free(:,it-1) + 0*B*ut_opt(:,it-1) + C);
end

end

