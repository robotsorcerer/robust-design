%% parameters
dt = 0.001;
tf = 100.0;

n_int = 100; 
xn = 6; 
un = 2; 
wn = 3; 

Q = 2*eye(xn);
R = eye(un);
gamma = 1;

%% data collection phase
x0 = normrnd(0,1,[6,1]);

[X, U] = int_system(dt, tf, x0);

n_data = floor(length(U)/n_int); 

Diff_xx = zeros(n_data, xn^2);
Int_xx = zeros(n_data, xn^2);
Int_ux = zeros(n_data, xn*un);

for i = 0:n_data-1
    diff_xx = kron(X((i+1)*n_int+1,:),X((i+1)*n_int+1,:)) ...
        - kron(X(i*n_int+1,:),X(i*n_int+1,:));
    int_xx = zeros(1,xn^2);
    int_ux = zeros(1,xn*un);
    
    for j = 1:n_int
        int_xx = int_xx + kron(X(i*n_int+j,:), X(i*n_int+j,:))*dt;
        int_ux = int_ux + kron(U(i*n_int+j,:), X(i*n_int+j,:))*dt;
    end
    
    Diff_xx(i+1,:) = diff_xx;  
    Int_xx(i+1,:) = int_xx;
    Int_ux(i+1,:) = int_ux;
end

%% Inner-loop test
I = 20;
J = 20;

A = [zeros(3), eye(3);
    12.54 -8.26 -0.39 -0.043 2.75 -0.36;
    -4.38 36.95 -3.00 0.086 -9.57 2.29;
    -6.82 -22.94 11.93 -0.034 6.82 -2.86;];

B = [zeros(3,2);
    -50.0 6.12;
    174.4 -38.93;
    -124.2 48.62;];

D = 0.1*[zeros(3,3);eye(3)];

[P, K, L] = solve_ARE(A, B, D, Q, R, gamma);

for i = 1:I
    L = zeros(wn, xn);
    for j = 1:J
        LD = kron(eye(xn),L'*D') + kron(L'*D',eye(xn));      
        Psi = Diff_xx + Int_xx*LD;
        
        KI = kron(K', eye(xn));
        Psi = [Psi, -2*(Int_ux + Int_xx*KI)];
        Psi = [Psi,  ones(n_data,1)];

%         [ZZ,RankFlag] = chol(Psi'*Psi);
%         RankFlag    

        Q_KL = -Q - K'*R*K + gamma^2*(L'*L);
        Reward = Int_xx*reshape(Q_KL,[xn^2,1]);

        pp = pinv(Psi)*Reward;

        P = pp(1:xn^2);
        P = reshape(P,[xn,xn]);
        P = (P+P')/2;
        L = gamma^(-2)*D'*P;
    end   
    K = pp(floor(xn^2+1):floor(xn^2 + xn*un));
    K = reshape(K, [xn,un]);
    K = R^(-1)*K';
end

   
