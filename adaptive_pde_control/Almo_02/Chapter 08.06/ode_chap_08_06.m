% Copyright (C) Henrik Anfinsen 2017-2019
%
% Feel free to use the code, but give credit to the original source and
% author

function dt = ode_chap_08_06(t, x, sys)

    
%% Parameter extraction

    N = sys.N;
    N_grid = sys.N_grid;
    
    lambda = sys.lambda;
    mu = sys.mu;
    c_1 = sys.c_1;
    c_2 = sys.c_2;
    
%% Variable extraction and augmentation
    dummy = reshape(x(1:(8*N)), N, 8);
    dummy_a = [2*dummy(1, :) - dummy(2, :); dummy; 2*dummy(N, :) - dummy(N-1, :)];
    
    u_sf_a  = dummy_a(:, 1);    u_sf = u_sf_a(2:(sys.N+1));
    v_sf_a  = dummy_a(:, 2);    v_sf = v_sf_a(2:(sys.N+1));
    u_of_a  = dummy_a(:, 3);    u_of = u_of_a(2:(sys.N+1));
    v_of_a  = dummy_a(:, 4);    v_of = v_of_a(2:(sys.N+1));
    u_hat_a = dummy_a(:, 5);    u_hat = u_hat_a(2:(sys.N+1));
    v_hat_a = dummy_a(:, 6);    v_hat = v_hat_a(2:(sys.N+1));
    u_tr_a  = dummy_a(:, 7);    u_tr = u_tr_a(2:(sys.N+1));
    v_tr_a  = dummy_a(:, 8);    v_tr = v_tr_a(2:(sys.N+1));

    
    U_sf_f = x(8*sys.N+1);
    U_of_f = x(8*sys.N+2);
    U_tr_f = x(8*sys.N+3);
    


%% Controller
    if (sys.ctrl_on == 1)
        U_sf = sys.intArr' * (sys.Kvu1 .* u_sf_a)   + sys.intArr' * (sys.Kvv1 .* v_sf_a);
        U_of = sys.intArr' * (sys.Kvu1 .* u_hat_a)  + sys.intArr' * (sys.Kvv1 .* v_hat_a);
        U_tr = sys.intArr' * (sys.Kvu1 .* u_tr_a)   + sys.intArr' * (sys.Kvv1 .* v_tr_a) ...
                    + sys.r_func(t + sys.t_2);
    else
        U_sf = 0;
        U_of = 0;
        U_tr = 0;
    end;
    
    
%% System and observer dynamics
    % Boundary conditions
    u_sf_a(1) = sys.q * v_sf_a(1);
    v_sf_a(sys.N_grid) = U_sf;
    
    u_of_a(1) = sys.q * v_of_a(1);
    v_of_a(sys.N_grid) = U_of;
    
    u_hat_a(1) = sys.q * v_hat_a(1);
    v_hat_a(sys.N_grid) = U_of;
    
    u_tr_a(1) = sys.q * v_tr_a(1);
    v_tr_a(sys.N_grid) = U_tr;
    
    % Spatial derivatives
    u_sf_x   = [(u_sf_a(2) - u_sf_a(1)) / sys.Delta; (u_sf_a(1:(N_grid-3)) - 4 * u_sf_a(2:(N_grid-2)) + 3*u_sf_a(3:(N_grid-1))) / (2 * sys.Delta)];
    v_sf_x   = [(- 3*v_sf_a(2:(N_grid-2)) + 4 * v_sf_a(3:(N_grid-1)) - v_sf_a(4:N_grid)) / (2 * sys.Delta); (v_sf_a(N_grid) - v_sf_a(N_grid-1)) / sys.Delta];
    
    u_of_x   = [(u_of_a(2) - u_of_a(1)) / sys.Delta; (u_of_a(1:(N_grid-3)) - 4 * u_of_a(2:(N_grid-2)) + 3*u_of_a(3:(N_grid-1))) / (2 * sys.Delta)];
    v_of_x   = [(- 3*v_of_a(2:(N_grid-2)) + 4 * v_of_a(3:(N_grid-1)) - v_of_a(4:N_grid)) / (2 * sys.Delta); (v_of_a(N_grid) - v_of_a(N_grid-1)) / sys.Delta];
   
    u_hat_x   = [(u_hat_a(2) - u_hat_a(1)) / sys.Delta; (u_hat_a(1:(N_grid-3)) - 4 * u_hat_a(2:(N_grid-2)) + 3*u_hat_a(3:(N_grid-1))) / (2 * sys.Delta)];
    v_hat_x   = [(- 3*v_hat_a(2:(N_grid-2)) + 4 * v_hat_a(3:(N_grid-1)) - v_hat_a(4:N_grid)) / (2 * sys.Delta); (v_hat_a(N_grid) - v_hat_a(N_grid-1)) / sys.Delta];
    
    u_tr_x   = [(u_tr_a(2) - u_tr_a(1)) / sys.Delta; (u_tr_a(1:(N_grid-3)) - 4 * u_tr_a(2:(N_grid-2)) + 3*u_tr_a(3:(N_grid-1))) / (2 * sys.Delta)];
    v_tr_x   = [(- 3*v_tr_a(2:(N_grid-2)) + 4 * v_tr_a(3:(N_grid-1)) - v_tr_a(4:N_grid)) / (2 * sys.Delta); (v_tr_a(N_grid) - v_tr_a(N_grid-1)) / sys.Delta];
   
    
    % Dynamics
    u_sf_t     = - lambda(2:(N+1)) .* u_sf_x + c_1(2:(N+1)) .* v_sf;
    v_sf_t     =       mu(2:(N+1)) .* v_sf_x + c_2(2:(N+1)) .* u_sf;
    
    u_of_t     = - lambda(2:(N+1)) .* u_of_x + c_1(2:(N+1)) .* v_of;
    v_of_t     =       mu(2:(N+1)) .* v_of_x + c_2(2:(N+1)) .* u_of;
    
    u_hat_t = - lambda(2:(N+1)) .* u_hat_x + c_1(2:(N+1)) .* v_hat ...
                    + sys.p1(2:(N+1)) * (u_of_a(end) - u_hat_a(end));
    v_hat_t =       mu(2:(N+1)) .* v_hat_x + c_2(2:(N+1)) .* u_hat ...
                    + sys.p2(2:(N+1)) * (u_of_a(end) - u_hat_a(end));
    
    u_tr_t     = - lambda(2:(N+1)) .* u_tr_x + c_1(2:(N+1)) .* v_tr;
    v_tr_t     =       mu(2:(N+1)) .* v_tr_x + c_2(2:(N+1)) .* u_tr;
    
    U_sf_f_t = sys.gamma_U * (U_sf - U_sf_f);
    U_of_f_t = sys.gamma_U * (U_of - U_of_f);
    U_tr_f_t = sys.gamma_U * (U_tr - U_tr_f);

    
%% Parse
    dt = [u_sf_t; v_sf_t; u_of_t; v_of_t; u_hat_t; v_hat_t; u_tr_t; v_tr_t; ...
            U_sf_f_t; U_of_f_t; U_tr_f_t];