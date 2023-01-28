% Copyright (C) Henrik Anfinsen 2017-2019
%
% Feel free to use the code, but give credit to the original source and
% author

function dt = ode_chap_10_04_01(t, x, sys)

    persistent last_printed_time;    

    if (isempty(last_printed_time) || (t > last_printed_time + 1))
        last_printed_time = t;
        disp(100 * t / sys.simH);
    end;
    
%% Parameter extraction

    N = sys.N;
    N_grid = sys.N_grid;
    
    lambda = sys.lambda;
    mu = sys.mu;
    c_1 = sys.c_1;
    c_2 = sys.c_2;
    
%% Variable extraction and augmentation
    dummy = reshape(x(1:(6*N)), N, 6);
    dummy_a = [2*dummy(1, :) - dummy(2, :); dummy; 2*dummy(N, :) - dummy(N-1, :)];
    
    u_a  = dummy_a(:, 1);    u = u_a(2:(sys.N+1));
    v_a  = dummy_a(:, 2);    v = v_a(2:(sys.N+1));
    eta_a  = dummy_a(:, 3);  eta = eta_a(2:(sys.N+1));
    phi_a  = dummy_a(:, 4);  phi = phi_a(2:(sys.N+1));
    p_a = dummy_a(:, 5);     p = p_a(2:(sys.N+1));
    r_a = dummy_a(:, 6);     r = r_a(2:(sys.N+1));

    q_hat = x(6*sys.N+1);
    U_f = x(6*sys.N+2);

    u_hat_a = eta_a + q_hat * p_a;
    v_hat_a = phi_a + q_hat * r_a;


%% Controller
    if (sys.ctrl_on == 1)
        params2 = sys.params;
        params2.q = q_hat;
        [Kvu_hat, Kvv_hat] = solver_2x2(params2, sys.N_grid);
        
        Kvu_hat1 = Kvu_hat(:, end);
        Kvv_hat1 = Kvv_hat(:, end);
        
        U = sys.intArr' * (Kvu_hat1 .* u_hat_a) + sys.intArr' * (Kvv_hat1 .* v_hat_a);
    else
        U = sys.U_func(t);
    end;
    
    
%% System and observer dynamics
    % Boundary conditions
    u_a(1) = sys.q * v_a(1);
    v_a(sys.N_grid) = U;
    
    eta_a(1) = 0;
    phi_a(sys.N_grid) = U;
    
    p_a(1) = v_a(1);
    r_a(sys.N_grid) = 0;
    
    % Spatial derivatives
    u_x   = [(u_a(2) - u_a(1)) / sys.Delta; (u_a(1:(N_grid-3)) - 4 * u_a(2:(N_grid-2)) + 3*u_a(3:(N_grid-1))) / (2 * sys.Delta)];
    v_x   = [(- 3*v_a(2:(N_grid-2)) + 4 * v_a(3:(N_grid-1)) - v_a(4:N_grid)) / (2 * sys.Delta); (v_a(N_grid) - v_a(N_grid-1)) / sys.Delta];
    
    eta_x   = [(eta_a(2) - eta_a(1)) / sys.Delta; (eta_a(1:(N_grid-3)) - 4 * eta_a(2:(N_grid-2)) + 3*eta_a(3:(N_grid-1))) / (2 * sys.Delta)];
    phi_x   = [(- 3*phi_a(2:(N_grid-2)) + 4 * phi_a(3:(N_grid-1)) - phi_a(4:N_grid)) / (2 * sys.Delta); (phi_a(N_grid) - phi_a(N_grid-1)) / sys.Delta];
   
    p_x     = [(p_a(2) - p_a(1)) / sys.Delta; (p_a(1:(N_grid-3)) - 4 * p_a(2:(N_grid-2)) + 3*p_a(3:(N_grid-1))) / (2 * sys.Delta)];
    r_x     = [(- 3*r_a(2:(N_grid-2)) + 4 * r_a(3:(N_grid-1)) - r_a(4:N_grid)) / (2 * sys.Delta); (r_a(N_grid) - r_a(N_grid-1)) / sys.Delta];
   
    
    % Dynamics
    u_t     = - lambda(2:(N+1)) .* u_x + c_1(2:(N+1)) .* v;
    v_t     =       mu(2:(N+1)) .* v_x + c_2(2:(N+1)) .* u;
    
    eta_t   = - lambda(2:(N+1)) .* eta_x + c_1(2:(N+1)) .* phi ...
                        + sys.p1(2:(N+1)) * (v_a(1) - phi_a(1));
    phi_t   =       mu(2:(N+1)) .* phi_x + c_2(2:(N+1)) .* eta ...
                        + sys.p2(2:(N+1)) * (v_a(1) - phi_a(1));
    
    p_t     = - lambda(2:(N+1)) .* p_x + c_1(2:(N+1)) .* r ...
                    - sys.p1(2:(N+1)) * r_a(1);
    r_t     =       mu(2:(N+1)) .* r_x + c_2(2:(N+1)) .* p ...
                    - sys.p2(2:(N+1)) * r_a(1);
    
    U_f_t = sys.gamma_U * (U - U_f);
    
    % Update law
        
    q_hat_t = sys.gamma * (v_a(1) - v_hat_a(1)) * r_a(1) / (1 + r_a(1)^2);


    
%% Parse
    dt = [u_t; v_t; eta_t; phi_t; p_t; r_t; ...
            q_hat_t; U_f_t];