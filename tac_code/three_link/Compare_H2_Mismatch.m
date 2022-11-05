clear all
clear 
clc

dt = 0.001;
t_f = 20;

A = [zeros(3), eye(3);
    12.54 -8.26 -0.39 -0.043 2.75 -0.36;
    -4.38 36.95 -3.00 0.086 -9.57 2.29;
    -6.82 -22.94 11.93 -0.034 6.82 -2.86;];

B = [zeros(3,2);
    -50.0 6.12;
    174.4 -38.93;
    -124.2 48.62;];

D = [zeros(3,3);eye(3)];

C = [eye(6);zeros(2,6)];
E = [zeros(6,2);5*eye(2)];
gamma = 12;

[K_lqr,~,~] = lqr(A,B,C'*C,E'*E);
[P_inf,K_inf,L_inf] = solve_ARE(A, B, D, C'*C, E'*E, gamma);

sys_lqr = ss(A-B*K_lqr,D,C-E*K_lqr,zeros(8,3));
sys_tf_lqr = tf(sys_lqr);
[ninf_lqr,~] = hinfnorm(sys_tf_lqr);

sys_inf = ss(A-B*K_inf,D,C-E*K_inf,zeros(8,3));
sys_tf_inf = tf(sys_inf);
[ninf_inf,~] = hinfnorm(sys_tf_inf);

X_inf = zeros(floor(t_f/dt+1),6);
X_lqr = zeros(floor(t_f/dt+1),6);
X_inf(1,:) = [0,-5,10,10,-10,-10];
X_lqr(1,:) = [0,-5,10,10,-10,-10];

DA = normrnd(0,1,[3,6]);
DB = normrnd(0,1,[3,2]);
% DA = [1, -1, 0, 0; -1, 1, 0, 0];
% DB = [-1; 1];
DPlt = [DA, 5*DB]/norm([DA, 5*DB]) * (0.8);

for step = 1:floor(t_f/dt)
    x = X_lqr(step,:)';
    w = DPlt*[x; -K_lqr*x];
    x = x + (A*x - B*K_lqr*x)*dt + D*w*dt;
    X_lqr(step+1,:) = x';
end

for step = 1:floor(t_f/dt)
    x = X_inf(step,:)';
    w = DPlt*[x; -K_inf*x];
    x = x + (A*x - B*K_inf*x)*dt + D*w*dt;
    X_inf(step+1,:) = x';
end

figure(1)
plot(0:dt:t_f,X_inf(:,1:2),'LineWidth',1.2);
xlabel({'Time (s)'},'Interpreter','latex','fontweight','bold','FontSize',20)
ylabel({'$\theta$ (deg) '},'Interpreter','latex','fontweight','bold','FontSize',20)
legend('$\theta_1$', '$\theta_2$', 'Interpreter','latex','fontweight','bold','FontSize',20)
set(gca,'FontSize',20)
set(gca,'XLim',[0,t_f])
set(gca,'YLim',[-50,100])
grid on

figure(2)
plot(0:dt:t_f,X_inf(:,3:4),'LineWidth',1.2);
xlabel({'Time (s)'},'Interpreter','latex','fontweight','bold','FontSize',20)
ylabel({'$\dot{\theta}$ (deg/s) '},'Interpreter','latex','fontweight','bold','FontSize',20)
legend('$\dot{\theta}_1$', '$\dot{\theta}_2$','Interpreter','latex','fontweight','bold', 'FontSize',20)
set(gca,'FontSize',20)
set(gca,'XLim',[0,t_f])
set(gca,'YLim',[-20,50])
grid on

figure(3)
plot(0:dt:t_f,X_lqr(:,1:2),'LineWidth',1.2);
xlabel({'Time (s)'},'Interpreter','latex','fontweight','bold','FontSize',20)
ylabel({'$\theta$ (deg) '},'Interpreter','latex','fontweight','bold','FontSize',20)
legend('$\theta_1$', '$\theta_2$', 'Interpreter','latex','fontweight','bold', 'FontSize',20)
set(gca,'FontSize',20)
set(gca,'XLim',[0,t_f])
set(gca,'YLim',[-20,50])
grid on

figure(4)
plot(0:dt:t_f,X_lqr(:,3:4),'LineWidth',1.2);
xlabel({'Time (s)'},'Interpreter','latex','fontweight','bold','FontSize',20)
ylabel({'$\dot{\theta}$ (deg/s) '},'Interpreter','latex','fontweight','bold','FontSize',20)
legend('$\dot{\theta}_1$', '$\dot{\theta}_2$','Interpreter','latex','fontweight','bold', 'FontSize',20)
set(gca,'FontSize',20)
set(gca,'XLim',[0,t_f])
set(gca,'YLim',[-50,100])
grid on
