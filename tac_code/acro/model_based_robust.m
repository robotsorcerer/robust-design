clear all
clc

A = [zeros(2), eye(2);
    12.49 -12.54 0 0;
    -14.49 29.36 0 0];

B = [0; 0; -2.98; 5.98];

D = [zeros(2,2); eye(2)];

C = [diag([1,1,1,1]);zeros(1,4)];
E = [zeros(4,1);1];

wn = 2;
xn = 4;
un = 1;

gamma = 35;
Q = C'*C;
R = E'*E;

K = [-897.7,-389.3,-391.0,-189.7];

P_pre = eye(6);
P = zeros(6,6);
step = 0;
I = 20;
J = 20;

K_store = zeros(I+1,un,xn);
K_store(1,:,:) = K;
P_store = zeros(I,I,xn,xn);
L_store = zeros(I,J,wn,xn);

for i = 1:I
    L = zeros(wn,xn);
    for j = 1:J
        QKL = Q + K'*R*K - gamma^2*(L')*L;
        AKL = A - B*K + D*L;
        P_pre = P;
        P = lyap(AKL',QKL);
     
        delta_L = normrnd(0,1,[3,6]);
        delta_L = 0.1*delta_L/norm(delta_L,'fro'); 
        L = gamma^(-2)*D'*P;
        
        L_store(i,j,:,:) = L;
        P_store(i,j,:,:) = P;
    end
    delta_K = normrnd(0,1,[un,xn]);
    delta_K = 0.6*delta_K/norm(delta_K,'fro'); 
    K = R^(-1)*B'*P + delta_K;
    K_store(i+1,:,:) = K;
end

[P_opt, K_opt, L_opt] = solve_ARE(A, B, D, Q, R, gamma);

Knorm = zeros(I+1,1);
for i=1:I+1
    Knorm(i) = norm(reshape(K_store(i,:,:),[un,xn])-K_opt, "fro")/norm(K_opt, "fro");
end
figure(1)
plot(1:I+1,Knorm,'-o','LineWidth',1.3)
xlabel({'\bf Iterations'},'Interpreter','latex', 'FontSize',20)
ylabel({'\bf Error'},'Interpreter','latex','FontSize',20)
legend({'$||K_i - K^\star||_F/||K^\star||_F$'},'Interpreter','latex','FontSize',20)
set(gca,'FontSize',20)
grid on
xlim([1,I+1])
grid on

Pnorm = zeros(I,1);
for i=1:I
    Pnorm(i) = norm(reshape(P_store(i,end,:,:),[xn,xn])-P_opt, "fro")/norm(P_opt, "fro");
end
figure(2)
plot(1:I,Pnorm,'-o','LineWidth',1.3)
xlabel({'\bf Iterations'},'Interpreter','latex','FontSize',20)
ylabel({'\bf Error'},'Interpreter','latex','FontSize',20)
legend({'$||P_{K_i} - P^\star||_F/||P^\star||_F$'},'Interpreter','latex','FontSize',20)
set(gca,'FontSize',20)
xlim([1,I])
grid on

Hnorm = zeros(I+1,1);
for i=1:I+1
    Ki = reshape(K_store(i,:,:),[un,xn]);
    sys_lqr = ss(A-B*Ki,D,C-E*Ki,zeros(xn+un,wn));
    sys_tf_lqr = tf(sys_lqr);
    [ninf,~] = hinfnorm(sys_tf_lqr);
    Hnorm(i) = ninf;
end
figure(3)
plot(1:I+1,Hnorm,'-o','LineWidth',1.3)
xlabel({'\bf Iteration'},'Interpreter','latex','FontSize',20)
ylabel({'\bf H-infinity Norm'},'Interpreter','latex','FontSize',20)
legend({'$||\mathcal{T}(K_i)||_{\mathcal{H}_{\infty}}$'},'Interpreter','latex','FontSize',20)
set(gca,'FontSize',20)
xlim([1,I+1])
grid on

