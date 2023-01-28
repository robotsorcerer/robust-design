% Copyright (C) Henrik Anfinsen 2017-2019
%
% Feel free to use the code, but give credit to the original source and
% author

clear all;
close all;

sys.ctrl_on = 1;
sys.second_order = 1;
sys.RGB_color1 = [1 0 0];
sys.RGB_color2 = [0 0 1];
sys.RGB_color3 = [0 1 0];
sys.style1 = '-';
sys.style2 = '-.';
sys.style3 = '--';

% System parameters
fun.lambda_1 = @(x)(1*x + 1);
fun.lambda_2 = @(x)(sin(pi*x) + 2);
fun.mu = @(x)(exp(x));
fun.lambda_1_d = @(x)(0*x + 1);
fun.lambda_2_d = @(x)(pi*cos(pi*x) + 0);
fun.mu_d = @(x)(1*exp(x) + 0);
fun.sigma_11 = @(x)(0*x + 1);
fun.sigma_21 = @(x)(cosh(x) + 0);
fun.sigma_12 = @(x)(1*x + 0);
fun.sigma_22 = @(x)(-1*x + 1);
fun.omega_1 = @(x)(0*x + 1);
fun.omega_2 = @(x)(2*x + 1);
fun.varpi_1 = @(x)(2*x + 1);
fun.varpi_2 = @(x)(-1*x + 1);
fun.q_1 = -1;
fun.q_2 = -2;
fun.c_1 = 1;
fun.c_2 = -1;

sys.fun = fun;

% Simulation specific
sys.N       = 120;
sys.N_grid  = sys.N + 2;
sys.N_g     = 200;
sys.simH    = 4;
sys.h       = 0.01;
sys.Tspan = 0:sys.h:sys.simH;
 
sys.Delta = 1 / (sys.N + 1);
sys.xspan = linspace(0, 1, sys.N_grid)';
sys.xspanT = (sys.Delta:sys.Delta:(1 - sys.Delta))';

sys.intArr = sys.Delta * [0.5; ones(sys.N, 1); 0.5];


% Expanding
sys.lambda_1 = fun.lambda_1(sys.xspan);
sys.lambda_2 = fun.lambda_2(sys.xspan);
sys.mu = fun.mu(sys.xspan);
sys.sigma_11 = fun.sigma_11(sys.xspan);
sys.sigma_21 = fun.sigma_21(sys.xspan);
sys.sigma_12 = fun.sigma_12(sys.xspan);
sys.sigma_22 = fun.sigma_22(sys.xspan);
sys.omega_1 = fun.omega_1(sys.xspan);
sys.omega_2 = fun.omega_2(sys.xspan);
sys.varpi_1 = fun.varpi_1(sys.xspan);
sys.varpi_2 = fun.varpi_2(sys.xspan);
sys.q_1 = fun.q_1;
sys.q_2 = fun.q_2;
sys.c_1 = fun.c_1;
sys.c_2 = fun.c_2;

sys.gamma_U = 10;

sys.t_u1 = sys.intArr' * (1 ./ sys.lambda_1);
sys.t_u2 = sys.intArr' * (1 ./ sys.lambda_1);
sys.t_v  = sys.intArr' * (1 ./ sys.mu);

sys.t_F = sys.t_u1 + sys.t_v;

sys.r_func = @(t)(1 + 1* sin(2 * pi * t));

    
%% Controller
[K1, K2, K3] = K_solver_n1(fun, sys.N_g);

sys.K1_1 = pchip(linspace(0, 1, sys.N_g), K1(:, end), sys.xspan);
sys.K2_1 = pchip(linspace(0, 1, sys.N_g), K2(:, end), sys.xspan);
sys.K3_1 = pchip(linspace(0, 1, sys.N_g), K3(:, end), sys.xspan);


%% Observer
[M1, M2, M3] = M_solver_n1(fun, sys.N_g);
sys.p1_1 = sys.mu(1) * pchip(linspace(0, 1, sys.N_g), M1(1, :)', sys.xspan);
sys.p1_2 = sys.mu(1) * pchip(linspace(0, 1, sys.N_g), M2(1, :)', sys.xspan);
sys.p2   = sys.mu(1) * pchip(linspace(0, 1, sys.N_g), M3(1, :)', sys.xspan);


    
    
%% Initial conditions

u1_sf_0 = ones(sys.N, 1);
u2_sf_0 = exp(sys.xspanT);
v_sf_0 = sin(pi*sys.xspanT);
u1_of_0 = ones(sys.N, 1);
u2_of_0 = ones(sys.N, 1);
v_of_0 = sin(pi*sys.xspanT);
u1_hat_0 = zeros(sys.N, 1);
u2_hat_0 = zeros(sys.N, 1);
v_hat_0 = zeros(sys.N, 1);
u1_tr_0 = ones(sys.N, 1);
u2_tr_0 = ones(sys.N, 1);
v_tr_0 = sin(pi*sys.xspanT);
U_sf_f_0 = 0;
U_of_f_0 = 0;
U_tr_f_0 = 0;

x0 = [u1_sf_0; u2_sf_0; v_sf_0; ...
      u1_of_0; u2_of_0; v_of_0; ...
      u1_hat_0; u2_hat_0; v_hat_0; ...
      u1_tr_0; u2_tr_0; v_tr_0; ...
      U_sf_f_0; U_of_f_0; U_tr_f_0];

tic;
[t_log, x_log] = ode45(@(t, x) ode_chap_14_06(t, x, sys), sys.Tspan, x0);
toc;

%% Post-processing
numT = length(t_log);

xx = reshape(x_log(:, 1:(12*sys.N)), numT, sys.N, 12);
xx_a = zeros(numT, sys.N + 2, 12);
xx_a(:, 1, :) = 2*xx(:, 1, :) - xx(:, 2, :);
xx_a(:, sys.N+2, :) = 2*xx(:, sys.N, :) - xx(:, sys.N-1, :);
xx_a(:, 2:(sys.N+1), :) = xx;

u1_sf_a  = squeeze(xx_a(:, :, 1));    u1_sf = u1_sf_a(:, 2:(sys.N+1));
u2_sf_a  = squeeze(xx_a(:, :, 2));    u2_sf = u2_sf_a(:, 2:(sys.N+1));
v_sf_a   = squeeze(xx_a(:, :, 3));    v_sf = v_sf_a(:, 2:(sys.N+1));

u1_of_a  = squeeze(xx_a(:, :, 4));    u1_of = u1_sf_a(:, 2:(sys.N+1));
u2_of_a  = squeeze(xx_a(:, :, 5));    u2_of = u2_sf_a(:, 2:(sys.N+1));
v_of_a   = squeeze(xx_a(:, :, 6));    v_of = v_sf_a(:, 2:(sys.N+1));

u1_hat_a  = squeeze(xx_a(:, :, 7));   u1_hat = u1_hat_a(:, 2:(sys.N+1));
u2_hat_a  = squeeze(xx_a(:, :, 8));   u2_hat = u2_hat_a(:, 2:(sys.N+1));
v_hat_a   = squeeze(xx_a(:, :, 9));   v_hat = v_hat_a(:, 2:(sys.N+1));

u1_tr_a  = squeeze(xx_a(:, :, 10));    u1_tr = u1_tr_a(:, 2:(sys.N+1));
u2_tr_a  = squeeze(xx_a(:, :, 11));    u2_tr = u2_tr_a(:, 2:(sys.N+1));
v_tr_a   = squeeze(xx_a(:, :, 12));    v_tr = v_tr_a(:, 2:(sys.N+1));

U_sf_f = x_log(:, 12*sys.N + 1);
U_of_f = x_log(:, 12*sys.N + 2);
U_tr_f = x_log(:, 12*sys.N + 3);


%% Plotting
uv_sf_norm = sqrt(u1_sf_a.^2 * sys.intArr) ...
           + sqrt(u2_sf_a.^2 * sys.intArr) ...
           + sqrt(v_sf_a.^2 * sys.intArr);
       
uv_of_norm = sqrt(u1_of_a.^2 * sys.intArr) ...
           + sqrt(u2_of_a.^2 * sys.intArr) ...
           + sqrt(v_of_a.^2 * sys.intArr);
       
uv_err_norm = sqrt((u1_of_a - u1_hat_a).^2 * sys.intArr) ...
            + sqrt((u2_of_a - u2_hat_a).^2 * sys.intArr) ...
            + sqrt((v_of_a - v_hat_a).^2 * sys.intArr);
        
uv_tr_norm = sqrt(u1_tr_a.^2 * sys.intArr) ...
           + sqrt(u2_tr_a.^2 * sys.intArr) ...
           + sqrt(v_tr_a.^2 * sys.intArr);

%% Plotting
[X_arr, T_arr] = meshgrid(sys.xspan, t_log);


%%% STATES NORMS
    figure(1);
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, uv_sf_norm, sys.style1, 'Color', sys.RGB_color1, 'LineWidth', 2);
    plot(t_log, uv_of_norm, sys.style2, 'Color', sys.RGB_color2, 'LineWidth', 2);
    plot(t_log, uv_tr_norm, sys.style3, 'Color', sys.RGB_color3, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$||u|| + ||v||$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = [uv_sf_norm; uv_of_norm; uv_tr_norm];
    d = (max(dataBlock) - min(dataBlock));
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    
    
    % ESTIMATION ERROR
    figure(2);
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, uv_err_norm, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$||u - \hat u|| + ||v - \hat v||$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = uv_err_norm;
    d = (max(dataBlock) - min(dataBlock));
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    
    
%% Actuation
    figure(3);
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, U_sf_f, sys.style1, 'Color', sys.RGB_color1, 'LineWidth', 2);
    plot(t_log, U_of_f, sys.style2, 'Color', sys.RGB_color2, 'LineWidth', 2);
    plot(t_log, U_tr_f, sys.style3, 'Color', sys.RGB_color3, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$U$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = [U_sf_f; U_of_f; U_tr_f];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    
    
    
%% Tracking goal
    figure(4);
    plot(t_log, sys.r_func(t_log), 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, v_tr_a(:, 1), sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$ y_0 $ and $ r $', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = v_tr_a(:, 1);
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    
    
%% Controller gains
    figure(5);
    plot(sys.xspan, sys.K1_1, sys.style1, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold on;
    plot(sys.xspan, sys.K2_1, sys.style2, 'Color', sys.RGB_color2, 'LineWidth', 2);
    plot(sys.xspan, sys.K3_1, sys.style3, 'Color', sys.RGB_color3, 'LineWidth', 2);
    hold off;
    xlabel('Space');
    ylabel('Controller gains', 'interpreter','latex');
    V = axis; V(2) = 1; axis(V);
    dataBlock = [sys.K1_1; sys.K2_1; sys.K3_1];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    
    
%% Observer gains
    figure(6);
    plot(sys.xspan, sys.p1_1, sys.style1, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold on;
    plot(sys.xspan, sys.p1_2, sys.style2, 'Color', sys.RGB_color2, 'LineWidth', 2);
    plot(sys.xspan, sys.p2, sys.style3, 'Color', sys.RGB_color3, 'LineWidth', 2);
    hold off;
    xlabel('Space');
    ylabel('Observer gains', 'interpreter','latex');
    V = axis; V(2) = 1; axis(V);
    dataBlock = [sys.p1_1; sys.p1_2; sys.p2];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;