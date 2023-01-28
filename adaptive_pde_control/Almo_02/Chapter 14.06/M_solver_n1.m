% Copyright (C) Henrik Anfinsen 2017-2019
%
% Feel free to use the code, but give credit to the original source and
% author

function [M1, M2, M3] = M_solver_n1(fun, N_g)

    xspan = linspace(0, 1, N_g);
    
%     \eps_1(x) K_x - \lambda_1(y) K_y - a_1(x, y) K - a_2(x, y) L - a_3(x, y) M = 0
%     \eps_2(x) L_x - \lambda_2(y) L_y - b_1(x, y) K - b_2(x, y) L - b_3(x, y) M = 0
%        \mu(x) M_x +       \mu(y) M_y - c_1(x, y) K - c_2(x, y) L - c_3(x, y) M = 0
%                               K(x, x) = f_1(x)
%                               L(x, x) = f_2(x)
%   M(x, 0) - q_1 K(x, 0) - q_2 L(x, 0) = 0
% defined for 0 \leq \xi \leq x \leq 1,

    params.eps_1    = fun.mu(1 - xspan);
    params.eps_2    = fun.mu(1 - xspan);
    params.lambda_1 = fun.lambda_1(1 - xspan);
    params.lambda_2 = fun.lambda_2(1 - xspan);
    params.mu       = fun.mu(1 - xspan);
    params.a_1      = zeros(N_g);
    params.a_2      = zeros(N_g);
    params.a_3      = zeros(N_g);
    params.b_1      = zeros(N_g);
    params.b_2      = zeros(N_g);
    params.b_3      = zeros(N_g);
    params.c_1      = zeros(N_g);
    params.c_2      = zeros(N_g);
    params.c_3      = zeros(N_g);
    for i = 1:N_g
        for j = 1:N_g
            params.a_1(i, j)  = fun.mu_d(1 - xspan(j)) + fun.sigma_11(1 - xspan(i));
            params.a_2(i, j)  = fun.sigma_12(1 - xspan(i));
            params.a_3(i, j)  = fun.omega_1(1 - xspan(i));

            params.b_1(i, j)  = fun.sigma_21(1 - xspan(i));
            params.b_2(i, j)  = fun.mu_d(1 - xspan(j)) + fun.sigma_22(1 - xspan(i));
            params.b_3(i, j)  = fun.omega_2(1 - xspan(i));

            params.c_1(i, j)  = fun.varpi_1(1 - xspan(i));
            params.c_2(i, j)  = fun.varpi_2(1 - xspan(i));
            params.c_3(i, j)  = fun.mu_d(1 - xspan(j));
        end;
    end;
    params.f_1        = fun.omega_1(1 - xspan) ./ (fun.lambda_1(1 - xspan) + fun.mu(1 - xspan));
    params.f_2        = fun.omega_2(1 - xspan) ./ (fun.lambda_2(1 - xspan) + fun.mu(1 - xspan));
    params.q_1        = fun.c_1;
    params.q_2        = fun.c_2;
    
    [K1, K2, K3] = solver_n1(params, N_g);
    
    M1 = zeros(N_g);
    M2 = zeros(N_g);
    M3 = zeros(N_g);
    
    for j = 1:N_g
        for i = 1:j
            M1(i, j) = K1(N_g - j + 1, N_g - i + 1);
            M2(i, j) = K2(N_g - j + 1, N_g - i + 1);
            M3(i, j) = K3(N_g - j + 1, N_g - i + 1);
        end;
    end;