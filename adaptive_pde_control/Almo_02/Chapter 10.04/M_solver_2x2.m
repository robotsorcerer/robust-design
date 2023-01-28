% Copyright (C) Henrik Anfinsen 2017-2019
%
% Feel free to use the code, but give credit to the original source and
% author

function [Ma, Mb] = M_solver_2x2(fun, N_g)

    xspan = linspace(0, 1, N_g);

    params.mu       = fun.mu(1 - xspan);
    params.lambda   = fun.lambda(1 - xspan);
    params.a_1      = zeros(N_g);
    params.a_2      = zeros(N_g);
    params.b_1      = zeros(N_g);
    params.b_2      = zeros(N_g);
    for i = 1:N_g
        for j = 1:N_g
            params.a_1(i, j)  = fun.mu_d(1 - xspan(j));
            params.a_2(i, j)  = fun.c_1(1 - xspan(i));
            params.b_1(i, j)  = fun.c_2(1 - xspan(i));
            params.b_2(i, j)  = fun.mu_d(1 - xspan(j));
        end;
    end;
%     params.f        = fun.c_1(1 - xspan) ./ (fun.lambda(1 - xspan) + fun.mu(1 - xspan));
    params.f        = fun.c_1(xspan) ./ (fun.lambda(xspan) + fun.mu(xspan));
    params.q        = 0;

    [Kvu, Kvv] = solver_2x2(params, N_g);
    
    Ma = zeros(N_g);
    Mb = zeros(N_g);
    for i = 1:N_g
        for j = 1:N_g
            Ma(i, j) = Kvu(N_g - j + 1, N_g - i + 1);
            Mb(i, j) = Kvv(N_g - j + 1, N_g - i + 1);
        end;
    end;