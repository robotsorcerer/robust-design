clear all
clear 
clc

dt = 0.001;
t_f = 20;

A = [zeros(2), eye(2);
    12.49 -12.54 0 0;
    -14.49 29.36 0 0];

B = [0; 0; -2.98; 5.98];

D = [zeros(2,2); eye(2)];

C = [diag([1,1,1,1]);zeros(1,4)];
E = [zeros(4,1);1];
gamma = 30;

[K_lqr,~,~] = lqr(A,B,C'*C,E'*E);
[P_inf,K_inf,L_inf] = solve_ARE(A, B, D, C'*C, E'*E, gamma);

sys_lqr = ss(A-B*K_lqr,D,C-E*K_lqr,zeros(5,2));
sys_tf_lqr = tf(sys_lqr);
[ninf_lqr,~] = hinfnorm(sys_tf_lqr);

sys_inf = ss(A-B*K_inf,D,C-E*K_inf,zeros(5,2));
sys_tf_inf = tf(sys_inf);
[ninf_inf,~] = hinfnorm(sys_tf_inf);

X_inf = zeros(floor(t_f/dt+1),4);
X_lqr = zeros(floor(t_f/dt+1),4);
X_inf(1,:) = [0,-5,5,5];
X_lqr(1,:) = [0,-5,5,5];

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
plot(0:dt:t_f,X_inf(:,1:2),'LineWidth',1.2);
xlabel({'\bf Time (s)'},'Interpreter','latex','fontweight','bold','FontSize',20)
ylabel({'$\mathbf{\theta}$ (deg) '},'Interpreter','latex','fontweight','bold','FontSize',20)
legend('$\mathbf{\theta_1}$', '$\mathbf{\theta_2}$', 'Interpreter','latex','fontweight','bold','FontSize',20)
set(gca,'FontSize',20)
set(gca,'XLim',[0,t_f])
set(gca,'YLim',[-50,100])
grid on

figure(2)
plot(0:dt:t_f,X_inf(:,3:4),'LineWidth',1.2);
xlabel({'\bf Time (s)'},'Interpreter','latex','fontweight','bold','FontSize',20)
ylabel({'$\mathbf{\dot{\theta}}$ (deg/s) '},'Interpreter','latex','fontweight','bold','FontSize',20)
legend('$\mathbf{\dot{\theta}_1}$', '$\mathbf{\dot{\theta}_2}$','Interpreter','latex','fontweight','bold', 'FontSize',20)
set(gca,'FontSize',20)
set(gca,'XLim',[0,t_f])
set(gca,'YLim',[-200,300])
grid on

figure(3)
plot(0:dt:0.5,X_lqr(1:501,1:2),'LineWidth',1.2);
xlabel({'\bf Time (s)'},'Interpreter','latex','fontweight','bold','FontSize',20)
ylabel({'$\mathbf{\theta}$ (deg) '},'Interpreter','latex','fontweight','bold','FontSize',20)
legend('$\mathbf{\theta_1}$', '$\mathbf{\theta_2}$', 'Interpreter','latex','fontweight','bold', 'FontSize',20)
set(gca,'FontSize',20)
set(gca,'XLim',[0,0.51])
set(gca,'YLim',[-50,100])
grid on

figure(4)
plot(0:dt:0.5,X_lqr(1:501,3:4),'LineWidth',1.2);
xlabel({'\bf Time (s)'},'Interpreter','latex','fontweight','bold','FontSize',20)
ylabel({'$\mathbf{\dot{\theta}}$ (deg/s) '},'Interpreter','latex','fontweight','bold','FontSize',20)
legend('$\mathbf{\dot{\theta}_1}$', '$\mathbf{\dot{\theta}_2}$','Interpreter','latex','fontweight','bold', 'FontSize',20)
set(gca,'FontSize',20)
set(gca,'XLim',[0, 0.51])
set(gca,'YLim',[-200,300])
grid on