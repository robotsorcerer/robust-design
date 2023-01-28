% Copyright (C) Henrik Anfinsen 2017-2019
%
% Feel free to use the code, but give credit to the original source and
% author

clear all;

sys.ctrl_on = 1;
sys.second_order = 1;
sys.RGB_color1 = [1 0 0];
sys.RGB_color2 = [0 0 1];
sys.RGB_color3 = [0 1 0];
sys.style1 = '-';
sys.style2 = '-.';
sys.style3 = '--';

% System parameters
fun.lambda = @(x)(2 + x .* cos(pi * x));
fun.mu = @(x)(3 - 2 * x);
fun.lambda_d = @(x)(cos(pi*x) - pi * x * sin(pi * x) + 0);
fun.mu_d = @(x)(0*x - 2);
fun.c_1 = @(x)(1 + 0*x);
fun.c_2 = @(x)(2 - x);
fun.q = 2;

sys.U_func = @(t)(0 * t + 0);


sys.fun = fun;

% Simulation specific
sys.N       = 20;
sys.N_grid  = sys.N + 2;
sys.N_g     = 100;
sys.simH    = 10;
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

sys.t_F = sys.intArr' * sys.lambda.^(-1) + sys.intArr' * sys.mu.^(-1);
sys.t_2 = sys.intArr' * (1 ./ sys.mu);

sys.gamma = 1;

sys.gamma_U = 10;

    
%% Controller
    params.mu       = fun.mu(sys.xspan);
    params.lambda   = fun.lambda(sys.xspan);
    params.a_1      = zeros(sys.N_grid);
    params.a_2      = zeros(sys.N_grid);
    params.b_1      = zeros(sys.N_grid);
    params.b_2      = zeros(sys.N_grid);
    for i = 1:sys.N_grid
        for j = 1:sys.N_grid
            params.a_1(i, j)  = fun.lambda_d(sys.xspan(i));
            params.a_2(i, j)  = fun.c_2(sys.xspan(i));
            params.b_1(i, j)  = fun.c_1(sys.xspan(i));
            params.b_2(i, j)  = - fun.mu_d(sys.xspan(i));
        end;
    end;
    params.f        = - fun.c_2(sys.xspan) ./ (fun.lambda(sys.xspan) + fun.mu(sys.xspan));
    params.q        = fun.q * fun.lambda(0) / fun.mu(0);

    sys.params = params;
    [Kvu, Kvv] = solver_2x2(params, sys.N_grid);
    
    
    sys.Kvu = Kvu;
    sys.Kvv = Kvv;
    
    sys.Kvu1 = Kvu(:, end);
    sys.Kvv1 = Kvv(:, end);
    
%% Observer gains
    [Ma, Mb] = M_solver_2x2(fun, sys.N_g);
    sys.Ma = Ma;
    sys.Mb = Mb;
    
    sys.p1 = sys.mu(1) * pchip(linspace(0, 1, sys.N_g)', Ma(1, :)', sys.xspan);
    sys.p2 = sys.mu(1) * pchip(linspace(0, 1, sys.N_g)', Mb(1, :)', sys.xspan);
    
%% Initial conditions

u_0 = 0 * ones(sys.N, 1);
v_0 = 1 + 0*sin(sys.xspanT);
eta_0 = zeros(sys.N, 1);
phi_0 = zeros(sys.N, 1);
p_0 = zeros(sys.N, 1);
r_0 = zeros(sys.N, 1);
q_hat_0 = 0;
U_f_0 = 0;

x0 = [u_0; v_0; eta_0; phi_0; p_0; r_0; q_hat_0; U_f_0];

tic;
[t_log, x_log] = ode45(@(t, x) ode_chap_10_04_01(t, x, sys), sys.Tspan, x0);
toc;

%% Post-processing
numT = length(t_log);

xx = reshape(x_log(:, 1:(6*sys.N)), numT, sys.N, 6);
xx_a = zeros(numT, sys.N + 2, 6);
xx_a(:, 1, :) = 2*xx(:, 1, :) - xx(:, 2, :);
xx_a(:, sys.N+2, :) = 2*xx(:, sys.N, :) - xx(:, sys.N-1, :);
xx_a(:, 2:(sys.N+1), :) = xx;

u_a  = squeeze(xx_a(:, :, 1));      u = u_a(:, 2:(sys.N+1));
v_a  = squeeze(xx_a(:, :, 2));      v = v_a(:, 2:(sys.N+1));
eta_a  = squeeze(xx_a(:, :, 3));    eta = eta_a(:, 2:(sys.N+1));
phi_a  = squeeze(xx_a(:, :, 4));    phi = phi_a(:, 2:(sys.N+1));
p_a = squeeze(xx_a(:, :, 5));       p = p_a(:, 2:(sys.N+1));
r_a = squeeze(xx_a(:, :, 6));       o = r_a(:, 2:(sys.N+1));

q_hat = x_log(:, 6*sys.N + 1); 
U_f = x_log(:, 6*sys.N + 2);



%% Plotting
u_hat_a = zeros(size(u_a));
v_hat_a = zeros(size(v_a));

for k = 1:numT
    u_hat_a(k, :) = eta_a(k, :) + q_hat(k) * p_a(k, :);
    v_hat_a(k, :) = phi_a(k, :) + q_hat(k) * r_a(k, :);
end;
uv_norm = sqrt(u_a.^2 * sys.intArr) + sqrt(v_a.^2 * sys.intArr);
error_norm = sqrt((u_a - u_hat_a).^2 * sys.intArr) + sqrt((v_a - v_hat_a).^2 * sys.intArr);

%% Plotting
[X_arr, T_arr] = meshgrid(sys.xspan, t_log);

figBx = 360;
figBy = 160;

    
  
%%% STATES NORM
    figure(1);
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, uv_norm, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$||u|| + ||v||$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    d = (max(uv_norm) - min(uv_norm));
    if (d > 0)
        V(3) = min(uv_norm) - 0.2*d;
        V(4) = max(uv_norm) + 0.2*d;
        axis(V);
    end;
    
  
%%% ESTIMATION ERROR
    figure(2);
    set(gcf, 'Color', [1 1 1], 'Units', 'pixels', 'Position', [2350,600,figBx,figBy]);
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, error_norm, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$||\hat e|| + ||\hat \epsilon||$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = error_norm;
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    
    
%% Actuation
    figure(3);
    set(gcf, 'Color', [1 1 1], 'Units', 'pixels', 'Position', [2000,300,figBx,figBy]);
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, U_f, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$U$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = U_f;
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    
  

%% Estimated paramer
    figure(4);
    set(gcf, 'Color', [1 1 1], 'Units', 'pixels', 'Position', [2350,300,figBx,figBy]);
    plot(t_log, t_log*0 + sys.q, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, q_hat, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$q$ and $\hat q$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = [q_hat; sys.q];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.5*d;
        V(4) = max(dataBlock) + 0.5*d;
        axis(V);
    end;
    