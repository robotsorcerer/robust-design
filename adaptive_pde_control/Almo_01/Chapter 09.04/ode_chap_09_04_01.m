% Copyright (C) Henrik Anfinsen 2018
%
% Feel free to reproduce the code, but give credit to the original source
% and author

function dt = ode_chap_09_04_01(t, x, sys)

    persistent last_printed_time;
    persistent last_kernel_update;
    persistent Ku1 Kv1;

    if (isempty(last_printed_time) || (t > last_printed_time + 1))
        last_printed_time = t;
        disp(100 * t / sys.simH);
    end;

%% Parameter extraction
    N = sys.N;
    N_grid = sys.N_grid;
    
    Delta = sys.Delta;
    
    lambda = sys.lambda;
    mu = sys.mu;
    c_1 = sys.c_1;
    c_2 = sys.c_2;
    c_3 = sys.c_3;
    c_4 = sys.c_4;
    q = sys.q;
    
    gamma1 = sys.gamma1;
    gamma2 = sys.gamma2;
    gamma3 = sys.gamma3;
    gamma4 = sys.gamma4;
    gamma5 = sys.gamma5;
    
    rho = sys.rho;
    
    k_U = sys.k_U;
    
%% Variable extraction and augmentation
    dummy = reshape(x(1:(4*N)), N, 4);
    dummy_a = [2*dummy(1, :) - dummy(2, :); dummy; 2*dummy(N, :) - dummy(N-1, :)];
    
    u_a = dummy_a(:, 1);        u = u_a(2:(N+1));
    v_a = dummy_a(:, 2);        v = v_a(2:(N+1));
    u_hat_a = dummy_a(:, 3);    u_hat = u_hat_a(2:(N+1));
    v_hat_a = dummy_a(:, 4);    v_hat = v_hat_a(2:(N+1));
    
    c1_hat = x(4*N+1);
    c2_hat = x(4*N+2);
    c3_hat = x(4*N+3);
    c4_hat = x(4*N+4);
    q_hat = x(4*N+5);
    
    U_f = x(4*N+6);
    
%% Calculations

    if (sys.ctrl_on == 1)
        if (isempty(last_kernel_update)) || (t >= last_kernel_update + sys.kernel_Update_Interval)
            last_kernel_update = t;
            fun.lambda = @(x)(0*x + sys.lambda);
            fun.mu = @(x)(0*x + sys.mu);
            fun.lambda_d = @(x)(0*x + 0);
            fun.mu_d = @(x)(0*x + 0);
            fun.c_1 = @(x)(c2_hat * exp(-(c1_hat - c4_hat) * x / sys.lambda));
            fun.c_2 = @(x)(c3_hat * exp( (c1_hat - c4_hat) * x / sys.lambda));
            fun.q = sys.q;

            [M, L] = K_solver_2x2(fun, sys.N_g);

            Ku1 = pchip(linspace(0, 1, sys.N_g), M(:, end), sys.xspan) ... 
                        ./ exp((c1_hat - c4_hat) * sys.xspan / sys.lambda);
            Kv1 = pchip(linspace(0, 1, sys.N_g), L(:, end), sys.xspan);
        end;
        U = (sys.intArr .* Ku1)' * u_hat_a + (sys.intArr .* Kv1)' * v_hat_a;
    else
        U = 0;
    end;
    
    
%% Boundary conditions
    % Compute
    u_0 = q * v_a(1);
    v_1 = U;
    
    u_hat_0 = (q_hat * v_a(1) + u_a(1) * v_a(1)^2) / (1 + v_a(1)^2);
    v_hat_1 = U;
    
    % Store values
    u_a(1) = u_0;
    v_a(N_grid) = v_1;
    
    u_hat_a(1) = u_hat_0;
    v_hat_a(end) = v_hat_1;
    
        
%% Spatial derivatives        
    u_x        = (u_a(2:(N+1)) - u_a(1:N)) / Delta;
    v_x        = (v_a(3:(N+2)) - v_a(2:(N+1))) / Delta;
    
    u_hat_x    = (u_hat_a(2:(N+1)) - u_hat_a(1:N)) / Delta;
    v_hat_x    = (v_hat_a(3:(N+2)) - v_hat_a(2:(N+1))) / Delta;
    
    
%% Dynamics
    % PDEs
    u_t     = - lambda * u_x + c_1 * u + c_2 * v;
    v_t     =       mu * v_x + c_3 * u + c_4 * v;
    
    u_hat_t = - lambda * u_hat_x + c1_hat * u + c2_hat * v ...
                + rho * (u - u_hat) * (u' * u) ...
                + rho * (u - u_hat) * (v' * v);
    v_hat_t =       mu * v_hat_x + c3_hat * u + c4_hat * v ...
                + rho * (v - v_hat) * (u' * u) ...
                + rho * (v - v_hat) * (v' * v);
    
    % ODEs
    c1_hat_t = gamma1 * sys.intArr' * (exp(- 0.01*sys.xspan) .* (u_a - u_hat_a) .* u_a);
    c2_hat_t = gamma2 * sys.intArr' * (exp(- 0.01*sys.xspan) .* (u_a - u_hat_a) .* v_a);
    c3_hat_t = gamma3 * sys.intArr' * (exp(  0.01*sys.xspan) .* (v_a - v_hat_a) .* u_a);
    c4_hat_t = gamma4 * sys.intArr' * (exp(  0.01*sys.xspan) .* (v_a - v_hat_a) .* v_a);

    q_hat_t = gamma5 * (u_a(1) - u_hat_a(1)) * v_a(1);


    % Projection
    if ((q_hat >= 5) && (q_hat_t > 0)) || ((q_hat <= 3) && (q_hat_t < 0))
        q_hat_t = 0;
    end;
    
    
    U_f_t = k_U * (U - U_f);
%% Parse
    dt = [u_t; v_t; u_hat_t; v_hat_t; c1_hat_t; c2_hat_t; c3_hat_t; c4_hat_t; q_hat_t; U_f_t];