function [P, K, L] = solve_ARE(A, B, D, Q, R, gamma)

    dt = 0.0001;
    xn = size(A,1);
    un = size(B,2);
    wu = size(D,2);
    P = zeros(xn,xn);
    P_pre = ones(xn,xn);

    step = 0;
    while norm(P_pre-P)>1e-8 && step<1e7
        step = step + 1;
        P_pre = P;
        P = P_pre + (A'*P_pre + P_pre*A + Q ...
            - P_pre*(B*R^(-1)*B' - gamma^(-2)*D*D')*P_pre)*dt;
    end

    K = R^(-1)*B'*P;
    L = gamma^(-2)*D'*P;
end