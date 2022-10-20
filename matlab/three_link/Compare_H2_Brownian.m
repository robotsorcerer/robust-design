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

% C = [diag([2,2,2,1,1,1]);zeros(2,6)];
% E = [zeros(6,2);10*eye(2)];
% gamma = 25;
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
    xi = normrnd(0,1,[3,1]);
    
    x_lqr = X_lqr(step,:)';
    x_lqr = x_lqr + (A*x_lqr - B*K_lqr*x_lqr)*dt + D*xi*sqrt(dt);
    X_lqr(step+1,:) = x_lqr';

    x_inf = X_inf(step,:)';
    x_inf = x_inf + (A*x_inf - B*K_inf*x_inf)*dt + D*xi*sqrt(dt);
    X_inf(step+1,:) = x_inf';
end

figure(1)
plot(0:dt:t_f,X_inf(:,1:3),'LineWidth',1.2);
xlabel({'\bf Time (s)'},'Interpreter','latex','FontSize',20)
ylabel({'$\mathbf{\theta}$ (deg) '},'Interpreter','latex','FontSize',20)
legend('$\mathbf{\theta_1}$', '$\mathbf{\theta_2}$', '$\mathbf{\theta_3}$', 'Interpreter','latex', 'FontSize',20)
set(gca,'FontSize',20)
set(gca,'XLim',[0,t_f])
set(gca,'YLim',[-15,20])
title('Joint Angles','Interpreter','latex','FontSize',20)
grid on
% set(gca,'Position',[0,0,500,500]);
set(gca, 'LooseInset', [0,0,0,0]);
saveas(gca, 'Plots/Brown_Hinf_ang_triple.eps', 'epsc')

figure(2)
plot(0:dt:t_f,X_inf(:,4:6),'LineWidth',1.2);
xlabel({'\bf Time (s)'},'Interpreter','latex','FontSize',20)
    ylabel({'$\mathbf{\dot{\theta}}$ (deg/s) '},'Interpreter','latex','FontSize',20)
legend('$\mathbf{\dot{\theta}_1}$', '$\mathbf{\dot{\theta}_2}$', '$\mathbf{\dot{\theta}_3}$','Interpreter','latex', 'FontSize',20)
set(gca,'FontSize',20)
set(gca,'XLim',[0,t_f])
title('Joint Velocities','Interpreter','latex','FontSize',20)
grid on
set(gca,'YLim',[-50,50])
% set (gca,'Position',[0,0,500,500]);
set(gca, 'LooseInset', [0,0,0,0]);
saveas(gca, 'Plots/Brown_Hinf_vel_triple.eps', 'epsc')

figure(3)
plot(0:dt:t_f,X_lqr(:,1:3),'LineWidth',1.2);
xlabel({'\bf Time (s)'},'Interpreter','latex','FontSize',20)
ylabel({'$\mathbf{\theta}$ (deg) '},'Interpreter','latex','FontSize',20)
legend('$\mathbf{\theta_1}$', '$\mathbf{\theta_2}$', '$\mathbf{\theta_3}$', 'Interpreter','latex', 'FontSize',20)
set(gca,'FontSize',20)
set(gca,'XLim',[0,t_f])
set(gca,'YLim',[-15,20])
grid on
title('Joint Angles','Interpreter','latex','FontSize',20)
% set (gca,'Position',[0,0,500,500]);
set(gca, 'LooseInset', [0,0,0,0]);
saveas(gca, 'Plots/Brown_LQG_ang_triple.eps', 'epsc')

figure(4)
plot(0:dt:t_f,X_lqr(:,4:6),'LineWidth',1.2);
xlabel({'\bf Time (s)'},'Interpreter','latex','FontSize',20)
ylabel({'$\mathbf{\dot{\theta}}$ (deg/s) '},'Interpreter','latex','FontSize',20)
legend('$\mathbf{\dot{\theta}_1}$', '$\mathbf{\dot{\theta}_2}$', '$\mathbf{\dot{\theta}_3}$','Interpreter','latex', 'FontSize',20)
set(gca,'FontSize',20)
set(gca,'XLim',[0,t_f])
set(gca,'YLim',[-50,50])
title('Joint Velocities','Interpreter','latex','FontSize',20)
grid on
set(gca, 'LooseInset', [0,0,0,0]);
saveas(gca, 'Plots/Brown_LQG_vel_triple.eps', 'epsc')
