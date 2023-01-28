% Copyright (C) Henrik Anfinsen 2017-2019
%
% Feel free to use the code, but give credit to the original source and
% author

function dt = ode_chap_06_04(t, x, sys)

    
    N = sys.N;
    N_grid = sys.N_grid;
    M = sys.M;
    M_grid = sys.M_grid;
    Delta = sys.Delta;
    
 
% Extraction of variables
    dummy1 = reshape(x(1:(5*N)), N, 5);
    dummy1_a = [2*dummy1(1, :) - dummy1(2, :); dummy1; 2*dummy1(N, :) - dummy1(N-1, :)];
    
    v_a = dummy1_a(:, 1);           v = v_a(2:(N_grid-1));
    psi_a = dummy1_a(:, 2);
    phi_a = dummy1_a(:, 3);         phi = phi_a(2:(N_grid-1));
    b_a = dummy1_a(:, 4);
    theta_hat_a = dummy1_a(:, 5);
    
    dummy2 = x((5*N+1):(5*N+M));
    dummy2_a = [2*dummy2(1, :) - dummy2(2, :); dummy2; 2*dummy2(M, :) - dummy2(M-1, :)];
    
    M2_a = dummy2_a;
    
    rho_hat = x(5*N + M + 1);
    r_f = x(5*N + M + 2);
    U_f = x(5*N + M + 3);
    
%% Calculations
    y = sys.k_2 * v_a(1);    

    
%% Control law
    r = sys.r_func(t);


    if (sys.ctrl_on == 1)
        z_hat_a = rho_hat *  psi_a - b_a;
        % First integral: theta and phi
        for i = 1:sys.N_grid
            for j = i:sys.N_grid
                z_hat_a(i) = z_hat_a(i) + sys.Delta * (1 - 0.5 * (j == i) - 0.5 * (j == sys.N_grid)) ...
                            * theta_hat_a(j) * phi_a(sys.N_grid - j + i);
            end;
        end;

        % Second integral: theta and M
        for i = 1:sys.N_grid
            x = Delta * (i - 1);
            for j = 1:sys.N_grid
                xi = Delta * (j - 1);

                val = 0.5 * (1 + x - xi);
                ind = round(val / sys.Delta_M) + 1;
                z_hat_a(i) = z_hat_a(i) + sys.Delta * (1 - 0.5 * (j == 1) - 0.5 * (j == sys.N_grid)) ...
                            * theta_hat_a(j) * M2_a(ind);
            end;
        end;

        numIter = 50;

        g_hat = zeros(sys.N_grid, 1);
        for k = 1:numIter
            conv_g_theta_hat = zeros(sys.N_grid, 1);

            for i = 2:sys.N_grid
                for j = 1:i
                    conv_g_theta_hat(i) = conv_g_theta_hat(i) + ...
                        sys.Delta * (1 - 0.5 * (j == 1) - 0.5 * (j == i)) * theta_hat_a(j) * g_hat(i - j + 1);
                end;
            end;
            g_hat = conv_g_theta_hat - theta_hat_a;
        end;
        
        U = (r - sys.intArr' * (theta_hat_a .* flipud(b_a)) ...
               + sys.intArr' * (flipud(g_hat) .* z_hat_a)) / rho_hat;
    else
        U = 0;
    end;
    
    
    
%% State augmentation
    v_a(N_grid) = sys.k_1 * U;
    psi_a(N_grid) = U;
    phi_a(N_grid) = y - b_a(1);
    b_a(N_grid) = r;
    
    M2_a(M_grid) = r;
    
    
%% Spatial derivatives
    if (sys.second_order == 1)
        v_x    = [(- 3*v_a(2:(N_grid-2)) + 4 * v_a(3:(N_grid-1)) - v_a(4:N_grid)) / (2 * sys.Delta); (v_a(N_grid) - v_a(N_grid-1)) / sys.Delta];
        psi_x  = [(- 3*psi_a(2:(N_grid-2)) + 4 * psi_a(3:(N_grid-1)) - psi_a(4:N_grid)) / (2 * sys.Delta); (psi_a(N_grid) - psi_a(N_grid-1)) / sys.Delta];
        phi_x  = [(- 3*phi_a(2:(N_grid-2)) + 4 * phi_a(3:(N_grid-1)) - phi_a(4:N_grid)) / (2 * sys.Delta); (phi_a(N_grid) - phi_a(N_grid-1)) / sys.Delta];
        b_x    = [(- 3*b_a(2:(N_grid-2)) + 4 * b_a(3:(N_grid-1)) - b_a(4:N_grid)) / (2 * sys.Delta); (b_a(N_grid) - b_a(N_grid-1)) / sys.Delta];

        M2_x   = [(- 3*M2_a(2:(M_grid-2)) + 4 * M2_a(3:(M_grid-1)) - M2_a(4:M_grid)) / (2 * sys.Delta_M); (M2_a(M_grid) - M2_a(M_grid-1)) / sys.Delta_M];
    else
        v_x    = (v_a(3:N_grid) - v_a(2:(N_grid-1))) / (sys.Delta);
        psi_x  = (psi_a(3:N_grid) - psi_a(2:(N_grid-1))) / (sys.Delta);
        phi_x  = (phi_a(3:N_grid) - phi_a(2:(N_grid-1))) / (sys.Delta);
        b_x    = (b_a(3:N_grid) - b_a(2:(N_grid-1))) / (sys.Delta);

        M2_x   = ( M2_a(3:M_grid) - M2_a(2:(M_grid-1))) / (sys.Delta_M);
    end;

    
%% Update law
    if (sys.ctrl_on == 1)
        val = 0.5 * (1 - sys.xspan);
        ind = round(val / sys.Delta_M) + 1;
        M20_a = M2_a(ind);
        M20 = M20_a(2:(N_grid-1));
        
        f_sq = psi_a(1)^2 + sys.intArr' * (phi_a.^2 + M20_a.^2);
        
        eps_hat_0 = y - b_a(1) - z_hat_a(1);
        
        rho_hat_t   = sys.gamma1 * eps_hat_0 * psi_a(1) / (1 + f_sq);
        theta_hat_t = sys.gamma2 * eps_hat_0 * (flipud(phi) + M20) / (1 + f_sq);
        
        if (((rho_hat <= sys.rho_hat_L) && (rho_hat_t < 0)) || ...
            ((rho_hat >= sys.rho_hat_U) && (rho_hat_t > 0)))
            rho_hat_t = 0;
        end;
    else
        rho_hat_t = 0;
        theta_hat_t = zeros(N, 1);
    end;

    

%% Dynamics
    v_t = sys.mu(2:(N_grid-1)) .* v_x + sys.f(2:(N_grid-1)) .* v ...
            + sys.g(2:(N_grid-1)) * v_a(1);
        for k = 1:N
            for i = 1:(k+1)
                v_t(k) = v_t(k) + sys.Delta * (1 - 0.5 * (i == 1) - 0.5 * (i == (k+1))) ...
                                    * sys.h_f(k, i) * v_a(i);
            end;
        end;
    psi_t = sys.mu_bar * psi_x;
    phi_t = sys.mu_bar * phi_x;
    b_t = sys.mu_bar * b_x;
    
    M2_t  = 0.5 * sys.mu_bar * M2_x;
    
    r_f_t = sys.k_U * (r - r_f);
    U_f_t = sys.k_U * (U - U_f);

    dt = [v_t; psi_t; phi_t; b_t; theta_hat_t; M2_t; rho_hat_t; r_f_t; U_f_t];