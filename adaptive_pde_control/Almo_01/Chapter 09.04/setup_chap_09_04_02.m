% Copyright (C) Henrik Anfinsen 2017-2019
%
% Feel free to use the code, but give credit to the original source and
% author


function [u_sti_w, u_sti_i, v_sti_w, v_sti_i] = setup_chap_09_04_02(sys)

    N = sys.N;

    u_sti_w = zeros(N, 3);
    u_sti_i = zeros(N, 3);
    
    v_sti_w = zeros(N, 3);
    v_sti_i = zeros(N, 3);
    
    u_sti_i(1, 1:3) = ones(1, 3);
    v_sti_i(N, 1:3) = N * ones(1, 3);

        % Interior
    % u1_x
    for k = 2:N
        if (k == 2) % First order upwind
            u_sti_w(k, 1:2) = -[1 -1] / (sys.Delta);
            u_sti_i(k, 1) = k;
            u_sti_i(k, 2) = k - 1;
            u_sti_i(k, 3) = k;
        else  % Second order upwind
            u_sti_w(k, 1:3) = -[3 -4 1] / (2 * sys.Delta);
            u_sti_i(k, 1) = k;
            u_sti_i(k, 2) = k - 1;
            u_sti_i(k, 3) = k - 2;
        end;
    end;

    % v_x
    for k = 1:(N-1)
        if (k == (N-1)) % First order upwind
            v_sti_w(k, 1:2) = -[1 -1] / (sys.Delta); 
            v_sti_i(k, 1) = k;
            v_sti_i(k, 2) = k + 1;
            v_sti_i(k, 3) = k;
        else % Second order upwind
            v_sti_w(k, 1:3) = -[3 -4 1] / (2 * sys.Delta); 
            v_sti_i(k, 1) = k;
            v_sti_i(k, 2) = k + 1;
            v_sti_i(k, 3) = k + 2;
        end;
    end;
