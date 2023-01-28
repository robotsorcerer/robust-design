% Copyright (C) Henrik Anfinsen 2018
%
% Feel free to reproduce the code, but give credit to the original source
% and author

function [Kvu, Kvv] = K_solver_2x2(fun, N_g)

    xspan = linspace(0, 1, N_g)';

    params.mu       = fun.mu(xspan);
    params.lambda   = fun.lambda(xspan);
    params.a_1      = zeros(N_g);
    params.a_2      = zeros(N_g);
    params.b_1      = zeros(N_g);
    params.b_2      = zeros(N_g);
    for i = 1:N_g
        for j = 1:N_g
            params.a_1(i, j)  = fun.lambda_d(xspan(i));
            params.a_2(i, j)  = fun.c_2(xspan(i));
            params.b_1(i, j)  = fun.c_1(xspan(i));
            params.b_2(i, j)  = - fun.mu_d(xspan(i));
        end;
    end;
    params.f        = - fun.c_2(xspan) ./ (fun.lambda(xspan) + fun.mu(xspan));
    params.q        = fun.q * fun.lambda(0) / fun.mu(0);

    [Kvu, Kvv] = solver_2x2(params, N_g);