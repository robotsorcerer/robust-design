I = 20;
J = 20;
K = [-32.55,-10.505,-4.4939,-10.485,-3.326,-2.10;
    -80.229,-29.07,-9.958,-26.161,-10.932,-4.433];
D = 0.1*[zeros(3,3);eye(3)];


for i = 1:I
    L = zeros(wn, xn);
    for j = 1:J
        Psi = [Diff_xx, 2*Int_xx*kron(L'*D', eye(xn))];
        KI = kron(K', eye(xn));
        Psi = [Psi, -2*(Int_ux - Int_xx*KI)];
        Psi = [Psi,  ones(n_data,1)];

        [ZZ,RankFlag] = chol(Psi'*Psi);
        RankFlag    

        Q_KL = -Q - K'*R*K + gamma^2*(L'*L);
        Reward = Int_xx*reshape(Q_KL,[xn^2,1]);

        pp = pinv(Psi)*Reward;

        P = pp(1:floor((xn+1)*xn/2));
        P = vecs_inv(P);
        L = gamma^(-2)*D'*P;
    end   
    K = pp(floor((xn+1)*xn/2 + xn^2 + 1):floor((xn+1)*xn/2 + xn^2 + xn*un));
    K = reshape(K, [xn,un]);
    K = K';
end
P = pp(1:floor((xn+1)*xn/2));
P = vecs_inv(P);