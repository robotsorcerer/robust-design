clc
clear all
%% parameters
dt = 0.0005;
tf = 1500.0;

xn = 4; 
un = 1; 
wn = 2; 
n_vecv = floor((xn+1)*xn/2);
n_phi = floor(n_vecv + xn*un + 1);

C = [diag([1,1,1,1]);zeros(1,4)];
E = [zeros(4,1);1];

gamma = 35;
Q = C'*C;
R = E'*E;

%% data collection phase
x0 = zeros(xn,1);
[X, U] = int_system(dt, tf, x0);

n_data = length(U); 

Phi = zeros(n_phi,n_phi);
Xi = zeros(n_phi, n_vecv);

parfor i = 1:n_data
    phi = [vecv(X(i,:)), 2*kron(X(i,:),U(i,:)), 1];
    Phi = Phi + phi'*phi*dt;
    Xi = Xi + phi'*(vecv(X(i+1,:)) - vecv(X(i,:)));
end
Phi = Phi/tf;
Xi = Xi/tf;
Phi_inv = pinv(Phi);

%% Learning-based algorithm
I = 20;
J = 30;

A = [zeros(2), eye(2);
    12.49 -12.54 0 0;
    -14.49 29.36 0 0];

B = [0; 0; -2.98; 5.98];

D = [zeros(2,2); eye(2)];

K = [-897.7,-389.3,-391.0,-189.7];

[T_v_vs, T_vs_v] = Trans_vec_vecs(xn);
T_xx_vecv = Trans_kron_vecv(xn);
T_vt = Trans_vec(un,xn);

K_store = zeros(I+1,un,xn);
K_store(1,:,:) = K;
P_store = zeros(I,I,xn,xn);
L_store = zeros(I,J,wn,xn);

tic
for i = 1:I
    L = zeros(wn, xn);
    for j = 1:J
        LD = kron(eye(xn),L'*D') + kron(L'*D',eye(xn));
        LD = T_v_vs*LD*T_vs_v;                
        KI = T_v_vs*(kron(eye(xn), K') + kron(K', eye(xn))*T_vt);        
        Q_KL = Q + K'*R*K - gamma^2*(L'*L);
        
        Lambda = Phi_inv(1:n_vecv,:)*Xi - KI*Phi_inv(n_vecv+1:n_vecv+xn*un,:)*Xi + LD;
         
        [ZZ,RankFlag] = chol(Lambda'*Lambda);
        RankFlag
        
        P_vecs = -pinv(Lambda)*vecs(Q_KL);   

        P = vecs_inv(P_vecs);
        L = gamma^(-2)*D'*P;
        L_store(i,j,:,:) = L;
        P_store(i,j,:,:) = P;
    end   
    BP = Phi_inv(n_vecv+1:n_vecv+xn*un,:)*Xi*P_vecs;
    BP = reshape(BP,[un,xn]);
    K = R^(-1)*BP;
    K_store(i+1,:,:) = K;
end
toc

[P_opt, K_opt, L_opt] = solve_ARE(A, B, D, Q, R, gamma);
Knorm = zeros(I+1,1);
for i=1:I+1
    Knorm(i) = norm(reshape(K_store(i,:,:),[un,xn])-K_opt, "fro")/norm(K_opt, "fro");
end
figure(1)
plot(1:I+1,Knorm,'-o','LineWidth',1.3)
xlabel({'Outer-loop Iteration'},'Interpreter','latex','FontSize',15)
ylabel({'Relative Distance'},'Interpreter','latex','FontSize',15)
legend({'$||K_i - K^\star||_F/||K^\star||_F$'},'Interpreter','latex','FontSize',15)
set(gca,'FontSize',15)
xlim([1,I+1])

Pnorm = zeros(I,1);
for i=1:I
    Pnorm(i) = norm(reshape(P_store(i,end,:,:),[xn,xn])-P_opt, "fro")/norm(P_opt, "fro");
end
figure(2)
plot(1:I,Pnorm,'-o','LineWidth',1.3)
xlabel({'Outer-loop Iteration'},'Interpreter','latex','FontSize',15)
ylabel({'Relative Distance'},'Interpreter','latex','FontSize',15)
legend({'$||P_{K_i} - P^\star||_F/||P^\star||_F$'},'Interpreter','latex','FontSize',15)
set(gca,'FontSize',15)
xlim([1,I])
