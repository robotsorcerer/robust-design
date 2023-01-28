% Copyright (C) Henrik Anfinsen 2017-2019
%
% Feel free to use the code, but give credit to the original source and
% author

function [K1, K2, K3] = K_solver_n1(fun, N_g)

    xspan = linspace(0, 1, N_g);

    params.eps_1    = fun.mu(xspan);
    params.eps_2    = fun.mu(xspan);
    params.lambda_1 = fun.lambda_1(xspan);
    params.lambda_2 = fun.lambda_2(xspan);
    params.mu       = fun.mu(xspan);
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
            params.a_1(i, j)  = fun.lambda_1_d(xspan(i)) + fun.sigma_11(xspan(i));
            params.a_2(i, j)  = fun.sigma_21(xspan(i));
            params.a_3(i, j)  = fun.varpi_1(xspan(i));

            params.b_1(i, j)  = fun.sigma_12(xspan(i));
            params.b_2(i, j)  = fun.lambda_2_d(xspan(i)) + fun.sigma_22(xspan(i));
            params.b_3(i, j)  = fun.varpi_2(xspan(i));

            params.c_1(i, j)  = fun.omega_1(xspan(i));
            params.c_2(i, j)  = fun.omega_2(xspan(i));
            params.c_3(i, j)  = - fun.mu_d(xspan(i));
        end;
    end;
    params.f_1        = - fun.varpi_1(xspan) ./ (fun.lambda_1(xspan) + fun.mu(xspan));
    params.f_2        = - fun.varpi_2(xspan) ./ (fun.lambda_2(xspan) + fun.mu(xspan));
    params.q_1        = fun.q_1 * fun.lambda_1(0) / fun.mu(0);
    params.q_2        = fun.q_2 * fun.lambda_2(0) / fun.mu(0);
    
    [K1, K2, K3] = solver_n1(params, N_g);