% Copyright (C) Henrik Anfinsen 2017-2019
%
% Feel free to use the code, but give credit to the original source and
% author

function dt = ode_chap_11_03(t, x, sys)

    persistent last_printed_time;

    if (isempty(last_printed_time) || (t > last_printed_time + 1))
        disp(100 * t / sys.simH);
        last_printed_time = t;
    end;
    
%% Parameter extraction

    N = sys.N;
    N_grid = sys.N_grid;
    
    lambda = sys.lambda;
    mu = sys.mu;
    
    lambda_bar = sys.lambda_bar;
    mu_bar = sys.mu_bar;
    c1 = sys.c1;
    c2 = sys.c2;
    q = sys.q;
    
    intArr = sys.intArr;
    
%% Variable extraction and augmentation
    dummy = reshape(x(1:(9*N)), N, 9);
    dummy_a = [2*dummy(1, :) - dummy(2, :); dummy; 2*dummy(N, :) - dummy(N-1, :)];
    
    u_a = dummy_a(:, 1);           u = u_a(2:(sys.N+1));
    v_a = dummy_a(:, 2);           v = v_a(2:(sys.N+1));
    psi_a = dummy_a(:, 3);         
    phi_a = dummy_a(:, 4);         phi = phi_a(2:(sys.N+1));
    P_a = dummy_a(:, 5);           
    p0_a = dummy_a(:, 6);          
    p1_a = dummy_a(:, 7);          p1 = p1_a(2:(sys.N+1));
    theta_hat_a = dummy_a(:, 8);   
    kappa_hat_a = dummy_a(:, 9);
    
    U_f = x(9*N + 1);
    
%% Compute state estimate
    z_hat_a = psi_a;
    for i = 1:sys.N_grid
        for j = i:sys.N_grid
            z_hat_a(i) = z_hat_a(i) + sys.Delta * (1 - 0.5 * (j == i) - 0.5 * (j == sys.N_grid)) ...
                        * theta_hat_a(j) * phi_a(sys.N_grid - (j - i));
        end;
    end;

    for i = 1:sys.N_grid
        for j = 1:sys.N_grid
            value = (sys.lambda_bar * (sys.N_grid - (i - 1)) + sys.mu_bar * (j - 1)) / (sys.lambda_bar + sys.mu_bar);
            ind = round(value);
            if (ind == 0)
                ind = 1;
            end;
            z_hat_a(i) = z_hat_a(i) + sys.Delta * (1 - 0.5 * (j == 1) - 0.5 * (j == sys.N_grid)) ...
                        * kappa_hat_a(j) * P_a(ind);
        end;
    end;
    
    
%% Controller
    if (sys.ctrl_on == 1)
        numIter = 30;

        g = theta_hat_a;
        for k = 1:numIter
            conv_g_theta_hat = zeros(sys.N_grid, 1);

            for i = 2:sys.N_grid
                for j = 1:i
                    conv_g_theta_hat(i) = conv_g_theta_hat(i) + ...
                        sys.Delta * (1 - 0.5 * (j == 1) - 0.5 * (j == i)) * theta_hat_a(j) * g(i - j + 1);
                end;
            end;
            g = conv_g_theta_hat - theta_hat_a;
        end;

        U = intArr' * (flipud(g) .* z_hat_a) - intArr' * (kappa_hat_a .* p1_a);
    else
        U = 0;
    end;
    
    
%% System and observer dynamics
    % Boundary conditions
    v_a(sys.N_grid) = U;
    u_a(1) = q * v_a(1);
    
    psi_a(sys.N_grid) = U;
    phi_a(sys.N_grid) = v_a(1);
    
    P_a(1) = v_a(1);
    p0_a(1) = phi_a(1);
    p1_a(1) = phi_a(sys.N_grid);

    % Spatial derivatives
    u_x   = [(u_a(2) - u_a(1)) / sys.Delta; (u_a(1:(N_grid-3)) - 4 * u_a(2:(N_grid-2)) + 3*u_a(3:(N_grid-1))) / (2 * sys.Delta)];
    v_x   = [(- 3*v_a(2:(N_grid-2)) + 4 * v_a(3:(N_grid-1)) - v_a(4:N_grid)) / (2 * sys.Delta); (v_a(N_grid) - v_a(N_grid-1)) / sys.Delta];
   
    psi_x   = [(- 3*psi_a(2:(N_grid-2)) + 4 * psi_a(3:(N_grid-1)) - psi_a(4:N_grid)) / (2 * sys.Delta); (psi_a(N_grid) - psi_a(N_grid-1)) / sys.Delta];
    phi_x   = [(- 3*phi_a(2:(N_grid-2)) + 4 * phi_a(3:(N_grid-1)) - phi_a(4:N_grid)) / (2 * sys.Delta); (phi_a(N_grid) - phi_a(N_grid-1)) / sys.Delta];
   
    P_x   = [(P_a(2) - P_a(1)) / sys.Delta; (P_a(1:(N_grid-3)) - 4 * P_a(2:(N_grid-2)) + 3*P_a(3:(N_grid-1))) / (2 * sys.Delta)];
    p0_x   = [(p0_a(2) - p0_a(1)) / sys.Delta; (p0_a(1:(N_grid-3)) - 4 * p0_a(2:(N_grid-2)) + 3*p0_a(3:(N_grid-1))) / (2 * sys.Delta)];
    p1_x   = [(p1_a(2) - p1_a(1)) / sys.Delta; (p1_a(1:(N_grid-3)) - 4 * p1_a(2:(N_grid-2)) + 3*p1_a(3:(N_grid-1))) / (2 * sys.Delta)];
    
    % Dynamics
    u_t     = - lambda(2:(N+1)) .* u_x + c1(2:(N+1)) .* v;
    v_t     =       mu(2:(N+1)) .* v_x + c2(2:(N+1)) .* u;
    
    psi_t   =  mu_bar * psi_x;
    phi_t   =  mu_bar * phi_x;
    
    P_t = - ((lambda_bar * mu_bar) / (lambda_bar + mu_bar)) * P_x;
    p0_t = - lambda_bar * p0_x;
    p1_t = - lambda_bar * p1_x;
   
    
%% Update law
    epsilon_hat_0 = v_a(1) - z_hat_a(1);
    theta_hat_t = sys.gamma1 * epsilon_hat_0 * flipud(phi) / (1 + phi_a' * phi_a + p1_a' * p1_a);
    kappa_hat_t = sys.gamma2 * epsilon_hat_0 * p1 / (1 + phi_a' * phi_a + p1_a' * p1_a);
    
    
%% Actuation signal storage
    U_f_t = sys.k_U * (U - U_f);
    
%% Parse
    dt = [u_t; v_t; psi_t; phi_t; P_t; p0_t; p1_t; theta_hat_t; kappa_hat_t; U_f_t];