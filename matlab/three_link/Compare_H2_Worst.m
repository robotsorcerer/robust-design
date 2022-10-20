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

C = [diag([1,1,1,1,1,1]);zeros(2,6)];
E = [zeros(6,2);1*eye(2)];
gamma = 5;

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

for step = 1:floor(t_f/dt)
    x = X_lqr(step,:)';
    x = x + (A*x - B*K_lqr*x + D*L_inf*x)*dt;
    X_lqr(step+1,:) = x';
end

for step = 1:floor(t_f/dt)
    x = X_inf(step,:)';
    x = x + (A*x - B*K_inf*x + D*L_inf*x)*dt;
    X_inf(step+1,:) = x';
end

figure(1)
plot(0:dt:t_f,X_inf(:,1:3),'LineWidth',1.2);
xlabel({'\bf Time (s)'},'Interpreter','latex','FontSize',20)
ylabel({'$\mathbf{\theta}$ (deg) '},'Interpreter','latex','FontSize',20)
legend('$\mathbf{\theta_1}$', '$\mathbf{\theta_2}$', '$\mathbf{\theta_3}$', 'Interpreter','latex', 'FontSize',20)
set(gca,'FontSize',20)
set(gca,'XLim',[0,t_f])
set(gca,'YLim',[-15,40])
title('Joint Angles','Interpreter','latex','FontSize',20)
grid on
% set(gca,'Position',[0,0,500,500]);
set(gca, 'LooseInset', [0,0,0,0]);
saveas(gca, 'Plots/Worst_Hinf_ang_triple.eps', 'epsc')

figure(2)
plot(0:dt:t_f,X_inf(:,4:6),'LineWidth',1.2);
xlabel({'\bf Time (s)'},'Interpreter','latex','FontSize',20)
    ylabel({'$\mathbf{\dot{\theta}}$ (deg/s) '},'Interpreter','latex','FontSize',20)
legend('$\mathbf{\dot{\theta}_1}$', '$\mathbf{\dot{\theta}_2}$', '$\mathbf{\dot{\theta}_3}$','Interpreter','latex', 'FontSize',20)
set(gca,'FontSize',20)
set(gca,'XLim',[0,t_f])
grid on
title('Joint Velocities','Interpreter','latex','FontSize',20)
set(gca,'YLim',[-100,150])
% set (gca,'Position',[0,0,500,500]);
set(gca, 'LooseInset', [0,0,0,0]);
saveas(gca, 'Plots/Worst_Hinf_vel_triple.eps', 'epsc')

figure(3)
plot(0:dt:1,X_lqr(1:1001,1:3),'LineWidth',1.2);
xlabel({'\bf Time (s)'},'Interpreter','latex','FontSize',20)
ylabel({'$\mathbf{\theta}$ (deg) '},'Interpreter','latex','FontSize',20)
legend('$\mathbf{\theta_1}$', '$\mathbf{\theta_2}$', '$\mathbf{\theta_3}$', 'Interpreter','latex', 'FontSize',20)
set(gca,'FontSize',20)
set(gca,'XLim',[0,1])
set(gca,'YLim',[-15,20])
title('Joint Angles','Interpreter','latex','FontSize',20)
grid on
% set (gca,'Position',[0,0,500,500]);
set(gca, 'LooseInset', [0,0,0,0]);
saveas(gca, 'Plots/Worst_LQG_ang_triple.eps', 'epsc')

figure(4)
plot(0:dt:1,X_lqr(1:1001,4:6),'LineWidth',1.2);
xlabel({'\bf Time (s)'},'Interpreter','latex','FontSize',20)
ylabel({'$\mathbf{\dot{\theta}}$ (deg/s) '},'Interpreter','latex','FontSize',20)
legend('$\mathbf{\dot{\theta}_1}$', '$\mathbf{\dot{\theta}_2}$', '$\mathbf{\dot{\theta}_3}$','Interpreter','latex', 'FontSize',20)
set(gca,'FontSize',20)
set(gca,'XLim',[0,1])
set(gca,'YLim',[-100,150])
title('Joint Velocities','Interpreter','latex','FontSize',20)
grid on
set(gca, 'LooseInset', [0,0,0,0]);
saveas(gca, 'Plots/Worst_LQG_vel_triple.eps', 'epsc')

