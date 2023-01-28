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
sys.style1 = '--';
sys.style2 = '-.';
sys.style3 = '-';

% System parameters
fun.lambda = @(x)(1*x + 1);
fun.mu = @(x)(1 * exp(x) + 1);
fun.lambda_d = @(x)(0*x + 1);
fun.mu_d = @(x)(1*exp(x) + 0);
fun.c_1 = @(x)(cosh(x) + 1);
fun.c_2 = @(x)(1*(x + 1));
fun.q = 2;

sys.fun = fun;

sys.r_func = @(t)(1 + sin(2 * pi *t));

% Simulation specific
sys.N       = 200;
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
sys.lambda = fun.lambda(sys.xspan);
sys.mu = fun.mu(sys.xspan);
sys.c_1 = fun.c_1(sys.xspan);
sys.c_2 = fun.c_2(sys.xspan);
sys.q = fun.q;

sys.t_1 = sys.intArr' * (1 ./ sys.lambda);
sys.t_2 = sys.intArr' * (1 ./ sys.mu);
sys.t_F = sys.t_1 + sys.t_2;

sys.gamma_U = 10;

    
%% Controller
[Kvu, Kvv] = K_solver_2x2(fun, sys.N_g);
sys.Kvu = Kvu;
sys.Kvv = Kvv;

sys.Kvu1 = pchip(linspace(0, 1, sys.N_g), Kvu(:, end), sys.xspan);
sys.Kvv1 = pchip(linspace(0, 1, sys.N_g), Kvv(:, end), sys.xspan);

%% Observer gains    
[Paa, Pba] = P_solver_2x2(fun, sys.N_g);
sys.Paa = Paa;
sys.Pba = Pba;

sys.p1 = - sys.lambda(end) * pchip(linspace(0, 1, sys.N_g)', Paa(end, :)', sys.xspan);
sys.p2 = - sys.lambda(end) * pchip(linspace(0, 1, sys.N_g)', Pba(end, :)', sys.xspan);
    
%% Initial conditions

u_sf_0 = ones(sys.N, 1);
v_sf_0 = sin(sys.xspanT);
u_of_0 = ones(sys.N, 1);
v_of_0 = sin(sys.xspanT);
u_hat_0 = zeros(sys.N, 1);
v_hat_0 = zeros(sys.N, 1);
u_tr_0 = ones(sys.N, 1);
v_tr_0 = sin(sys.xspanT);
U_sf_f_0 = 0;
U_of_f_0 = 0;
U_tr_f_0 = 0;

x0 = [u_sf_0; v_sf_0; u_of_0; v_of_0; u_hat_0; v_hat_0; u_tr_0; v_tr_0; ...
        U_sf_f_0; U_of_f_0; U_tr_f_0];

tic;
[t_log, x_log] = ode45(@(t, x) ode_chap_08_06(t, x, sys), sys.Tspan, x0);
toc;

%% Post-processing
numT = length(t_log);

xx = reshape(x_log(:, 1:(8*sys.N)), numT, sys.N, 8);
xx_a = zeros(numT, sys.N + 2, 8);
xx_a(:, 1, :) = 2*xx(:, 1, :) - xx(:, 2, :);
xx_a(:, sys.N+2, :) = 2*xx(:, sys.N, :) - xx(:, sys.N-1, :);
xx_a(:, 2:(sys.N+1), :) = xx;

u_sf_a  = squeeze(xx_a(:, :, 1));    u_sf = u_sf_a(:, 2:(sys.N+1));
v_sf_a  = squeeze(xx_a(:, :, 2));    v_sf = v_sf_a(:, 2:(sys.N+1));
u_of_a  = squeeze(xx_a(:, :, 3));    u_of = u_of_a(:, 2:(sys.N+1));
v_of_a  = squeeze(xx_a(:, :, 4));    v_of = v_of_a(:, 2:(sys.N+1));
u_hat_a = squeeze(xx_a(:, :, 5));    u_hat = u_hat_a(:, 2:(sys.N+1));
v_hat_a = squeeze(xx_a(:, :, 6));    v_hat = v_hat_a(:, 2:(sys.N+1));
u_tr_a  = squeeze(xx_a(:, :, 7));    u_tr = u_tr_a(:, 2:(sys.N+1));
v_tr_a  = squeeze(xx_a(:, :, 8));    v_tr = v_tr_a(:, 2:(sys.N+1));

U_sf_f = x_log(:, 8*sys.N + 1);
U_of_f = x_log(:, 8*sys.N + 2);
U_tr_f = x_log(:, 8*sys.N + 3);

U_sf = zeros(numT, 1);
U_of = zeros(numT, 1);
U_tr = zeros(numT, 1);

for k = 1:numT
    U_sf(k) = sys.intArr' * (sys.Kvu1 .* u_sf_a(k, :)')   + sys.intArr' * (sys.Kvv1 .* v_sf_a(k, :)');
    U_of(k) = sys.intArr' * (sys.Kvu1 .* u_of_a(k, :)')   + sys.intArr' * (sys.Kvv1 .* v_of_a(k, :)');
    U_tr(k) = sys.intArr' * (sys.Kvu1 .* u_tr_a(k, :)')   + sys.intArr' * (sys.Kvv1 .* v_tr_a(k, :)') ...
                    + sys.r_func(t_log(k));
end;



%% Plotting
uv_sf_norm = sqrt(u_sf_a.^2 * sys.intArr) + sqrt(v_sf_a.^2 * sys.intArr);
uv_of_norm = sqrt(u_of_a.^2 * sys.intArr) + sqrt(v_of_a.^2 * sys.intArr);
e_of_norm  = sqrt((u_of_a - u_hat_a).^2 * sys.intArr) + sqrt((v_of_a - v_hat_a).^2 * sys.intArr);
uv_tr_norm = sqrt(u_tr_a.^2 * sys.intArr) + sqrt(v_tr_a.^2 * sys.intArr);

%% Plotting
[X_arr, T_arr] = meshgrid(sys.xspan, t_log);


    %% COMBINED PLOT
    figure(1);
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, uv_sf_norm, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    plot(t_log, uv_of_norm, sys.style2, 'Color', sys.RGB_color2, 'LineWidth', 2);
    plot(t_log, uv_tr_norm, sys.style1, 'Color', sys.RGB_color3, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$||u|| + ||v||$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = [uv_sf_norm; uv_of_norm; uv_tr_norm];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    
    
    
    %% ESTIMATION ERROR
    figure(2);
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, e_of_norm, sys.style2, 'Color', sys.RGB_color2, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$||\tilde u|| + ||\tilde v||$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    d = (max(e_of_norm) - min(e_of_norm));
    if (d > 0)
        V(3) = min(e_of_norm) - 0.2*d;
        V(4) = max(e_of_norm) + 0.2*d;
        axis(V);
    end;
    
    
    
%% Actuation
    figure(3);
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, U_sf, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    plot(t_log, U_of, sys.style2, 'Color', sys.RGB_color2, 'LineWidth', 2);
    plot(t_log, U_tr, sys.style1, 'Color', sys.RGB_color3, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$U$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = [U_sf; U_of; U_tr];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    
    
%% Objective
    figure(4);
    plot(t_log, sys.r_func(t_log), 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, v_tr_a(:, 1), sys.style1, 'Color', sys.RGB_color3, 'LineWidth', 2);
    xlabel('Time [s]');
    ylabel('$y$ and $r$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = sys.r_func(t_log);
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.5*d;
        V(4) = max(dataBlock) + 0.5*d;
        axis(V);
    end;
    
    
%% Controller gain
    figure(5);
    plot(sys.xspan, sys.Kvu1, sys.style1, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold on;
    plot(sys.xspan, sys.Kvv1, sys.style2, 'Color', sys.RGB_color2, 'LineWidth', 2);
    hold off;
    xlabel('x');
    ylabel('Controller gains', 'interpreter','latex');
    V = axis; V(2) = 1; axis(V);
    dataBlock = [sys.Kvu1; sys.Kvv1];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;

    
%% Observer gain
    figure(6);
    plot(sys.xspan, sys.p1, sys.style1, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold on;
    plot(sys.xspan, sys.p2, sys.style2, 'Color', sys.RGB_color2, 'LineWidth', 2);
    hold off;
    xlabel('x');
    ylabel('Injection gains', 'interpreter','latex');
    V = axis; V(2) = 1; axis(V);
    dataBlock = [sys.p1; sys.p2];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;