% Copyright (C) Henrik Anfinsen 2017-2019
%
% Feel free to use the code, but give credit to the original source and
% author

function dt = ode_chap_09_04_02(t, x, sys)

    N = sys.N;
    
    persistent last_printed_time;
    persistent last_kernel_update;
    persistent Ku1 Kv1;

    if (isempty(last_printed_time) || (t > last_printed_time + 1))
        last_printed_time = t;
        disp(100 * t / sys.simHoriz);
    end;
    
    
%% Variable extraction
    dummy_states = reshape(x(1:(6*sys.N)), N, 6);
    u       = dummy_states(:,  1);
    v       = dummy_states(:,  2);
    eta     = dummy_states(:,  3);
    phi     = dummy_states(:,  4);
    theta_hat   = dummy_states(:,  5);
    kappa_hat   = dummy_states(:,  6);
    
    dummy_states2 = reshape(x((6*sys.N+1):(6*sys.N+2*N^2)), N, 2*N);
    M_filt  = dummy_states2(:,  1:N);
    N_filt  = dummy_states2(:,  (N+1):(2*N));
    
    q_hat = x(6*sys.N+2*N^2 + 1);
    
    u_sti_w = sys.u_sti_w;
    u_sti_i = sys.u_sti_i;
    v_sti_w = sys.v_sti_w;
    v_sti_i = sys.v_sti_i;
    u_i     = sys.u_i;
    v_i     = sys.v_i;
    
    lambda  = sys.lambda;
    mu      = sys.mu;
    
    
%% Controller
%     U = sin(3*t) + sin(3*0.5 * sqrt(3) * t) + sin(2*t) + sin(0.5 * pi * t);


    if (sys.ctrl_on == 1)
        
        if (isempty(last_kernel_update)) || (t >= last_kernel_update + sys.kernel_Update_Interval)
            last_kernel_update = t;
            fun.lambda = @(x)(0*x + sys.lambda);
            fun.mu = @(x)(0*x + sys.mu);
            fun.lambda_d = @(x)(0*x + 0);
            fun.mu_d = @(x)(0*x + 0);
            fun.c_1 = @(x)(sys.lambda * theta_hat(1 + round(x * (sys.N - 1))));
            fun.c_2 = @(x)(sys.mu * kappa_hat(1 + round(x * (sys.N - 1))));
            fun.q = sys.q;

            [K, L] = K_solver_2x2(fun, sys.N_g);

            Ku1 = pchip(linspace(0, 1, sys.N_g), K(:, end), sys.xspan);
            Kv1 = pchip(linspace(0, 1, sys.N_g), L(:, end), sys.xspan);
        end;



        U = (sys.intArr .* Ku1)' * u + (sys.intArr .* Kv1)' * v;


    else
        U = sys.U_func(t);
    end;
    

%% Dynamics
    % u/v states
        % Spatial derivative
        u_t = lambda * sum(u_sti_w .* u(u_sti_i), 2);
        v_t = mu * sum(v_sti_w .* v(v_sti_i), 2);

        % Source terms
        u_t(u_i) = u_t(u_i) + sys.c1(u_i) .* v(u_i);
        v_t(v_i) = v_t(v_i) + sys.c2(v_i) .* u(v_i);
        
        % BCs
        u_t(1) = sys.gamma_BC * (sys.q * v(1) - u(1));
        v_t(N)  = sys.gamma_BC * (U - v(N));
        
    % eta states
        % Spatial derivative
        eta_t = lambda * sum(u_sti_w .* eta(u_sti_i), 2);
        
        % BCs
        eta_t(1) = sys.gamma_BC * (v(1) - eta(1));
        
    % eta states
        % Spatial derivative
        phi_t = mu * sum(v_sti_w .* phi(v_sti_i), 2);
        
        % BCs
        phi_t(N) = sys.gamma_BC * (U - phi(N));
        
    % M and N??
        % Spatial derivatives
        M_filt_t = zeros(N);
        N_filt_t = zeros(N);
        for i = 1:N
            M_filt_i = M_filt(i, :)';
            M_filt_i_x = (M_filt_i(1:(N-1)) - M_filt_i(2:(N))) / sys.Delta;
            M_filt_t(i, (2:(N))) = lambda * M_filt_i_x;
            
            
            N_filt_i = N_filt(i, :)';
            N_filt_i_x = (N_filt_i(2:N) - N_filt_i(1:(N-1))) / sys.Delta;
            N_filt_t(i, (1:(N-1))) = mu * N_filt_i_x;
        end;
        
        % BCs
        M_filt_t(logical(eye(N))) = sys.gamma_BC * (v - diag(M_filt));
        N_filt_t(logical(eye(N))) = sys.gamma_BC * (u - diag(N_filt));
        
        
%% Adaptive laws

    u_hat = q_hat * eta;
    v_hat = phi;
    
    for i = 1:N
        for j = 1:i
            u_hat(i) = u_hat(i) + sys.Delta * (1 - 0.5*(j == 1) - 0.5*(j == i)) ...
                                    * theta_hat(j) * M_filt(j, i);
        end;
        
        for j = i:N
            v_hat(i) = v_hat(i) + sys.Delta * (1 - 0.5*(j == i) - 0.5*(j == N)) ...
                                    * kappa_hat(j) * N_filt(j, j);
        end;
    end;
    
    n_0 = N_filt(:, 1);
    
    f_sq = eta'* eta * sys.Delta + sum(sum(M_filt.^2)) * sys.Delta^2;
    N_norm_sq = sum(sum(N_filt.^2)) * sys.Delta^2;
    n_0_sq = n_0'* n_0 * sys.Delta;
    
    e_hat = u - u_hat;
    epsilon_hat = v - v_hat;

    q_hat_t     = sys.gamma0  * sys.intArr' * (e_hat .* eta) / (1 + f_sq);

    theta_hat_t = zeros(N, 1);
    for i = 1:N
        for j = i:N
            theta_hat_t(i) = theta_hat_t(i) + sys.Delta * (1 - 0.5*(j == i) - 0.5*(j == N)) ...
                                * e_hat(j) * M_filt(i, j);
        end;
    end;
    theta_hat_t = sys.gamma1 * theta_hat_t / (1 + f_sq);

    kappa_hat_t1 = zeros(N, 1);
    for i = 1:N
        for j = 1:i
            kappa_hat_t1(i) = kappa_hat_t1(i) + sys.Delta * (1 - 0.5*(j == 1) - 0.5*(j == i)) ...
                                * epsilon_hat(j) * N_filt(i, j);
        end;
    end;

    kappa_hat_t = sys.gamma2 * kappa_hat_t1 / (1 + N_norm_sq) ...
                    + sys.gamma2  * (epsilon_hat(1) * n_0) / (1 + n_0_sq);
    

%% Parse variables
dt = [u_t; v_t; ...
        eta_t; phi_t; ...
        theta_hat_t; kappa_hat_t; ...
        M_filt_t(:); N_filt_t(:); ...
        q_hat_t];