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

if (sys.ctrl_on == 1)
    sys.simH    = 15;
else
    sys.simH    = 10;
end;

% Simulation specific
sys.N       = 50;
sys.N_grid  = sys.N + 2;
sys.h       = 0.01;
sys.Tspan   = 0:sys.h:sys.simH;
 
sys.Delta = 1 / (sys.N + 1);
sys.xspan = (0:sys.Delta:1)';
sys.xspanT = (sys.Delta:sys.Delta:(1 - sys.Delta))';

sys.intArr = sys.Delta * [0.5; ones(sys.N, 1); 0.5];

% System parameters
sys.lambda = sys.xspan * 0.5 + 0.5;
sys.mu = exp(0.5*sys.xspan);
sys.lambda_bar = (sys.intArr' * (1 ./ sys.lambda))^(-1);
sys.mu_bar = (sys.intArr' * (1 ./ sys.mu))^(-1);

sys.c1 = sys.xspan * 1 + 1;
sys.c2 = sin(sys.xspan) + 1;
sys.q = 1;

% Tuning
sys.gamma1 = 100;
sys.gamma2 = 100;

sys.k_U = 100;
    
%% Initial conditions

    %% States
u0 = sys.xspanT;
v0 = sin(2*pi*sys.xspanT);

    %% Filters
psi0 = zeros(sys.N, 1);
phi0 = zeros(sys.N, 1);
P0 = zeros(sys.N, 1);
p0_0 = zeros(sys.N, 1);
p1_1 = zeros(sys.N, 1);
    
    %% Parameter estimates
theta_hat0 = 0*ones(sys.N, 1);
kappa_hat0 = 0*ones(sys.N, 1);

    %% Actuation
U_f0 = 0;

x0 = [u0; v0; psi0; phi0; P0; p0_0; p1_1; theta_hat0; kappa_hat0; U_f0];

tic;
[t_log, x_log] = ode45(@(t, x) ode_chap_11_03(t, x, sys), sys.Tspan, x0);
toc;

%% Post-processing
numT = length(t_log);

xx = reshape(x_log(:, 1:(9*sys.N)), numT, sys.N, 9);
xx_a = zeros(numT, sys.N + 2, 9);
xx_a(:, 1, :) = 2*xx(:, 1, :) - xx(:, 2, :);
xx_a(:, sys.N+2, :) = 2*xx(:, sys.N, :) - xx(:, sys.N-1, :);
xx_a(:, 2:(sys.N+1), :) = xx;

u_a = squeeze(xx_a(:, :, 1));           w = u_a(:, 2:(sys.N+1));
v_a = squeeze(xx_a(:, :, 2));           z = v_a(:, 2:(sys.N+1));
psi_a = squeeze(xx_a(:, :, 3));         psi = psi_a(:, 2:(sys.N+1));
phi_a = squeeze(xx_a(:, :, 4));         phi = phi_a(:, 2:(sys.N+1));
P_a = squeeze(xx_a(:, :, 5));           P = P_a(:, 2:(sys.N+1));
p0_a = squeeze(xx_a(:, :, 6));          p0 = p0_a(:, 2:(sys.N+1));
p1_a = squeeze(xx_a(:, :, 7));          p1 = p1_a(:, 2:(sys.N+1));
theta_hat_a = squeeze(xx_a(:, :, 8));   theta_hat = theta_hat_a(:, 2:(sys.N+1));
kappa_hat_a = squeeze(xx_a(:, :, 9));   kappa_hat = kappa_hat_a(:, 2:(sys.N+1));

U_f = x_log(:, 9*sys.N+1);

U_f(1) = 2*U_f(2) - U_f(3);

% Calculations
XSPAN = repmat(sys.xspan', numT, 1);
[X_arr, T_arr] = meshgrid(sys.xspan, t_log);

uv_norm = sqrt(u_a.^2 * sys.intArr) + sqrt(v_a.^2 * sys.intArr);


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
    
    
%% Actuation
    figure(2);
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
  
    
%% Estimated parameters
    figure(3);
    surf(T_arr, X_arr, theta_hat_a, 'EdgeColor', 'none');
    V = axis; V(2) = t_log(end); axis(V);
    xlabel('Time [s]');
    ylabel('Space');
    zlabel('$\hat \theta$', 'interpreter','latex');
    colormap(parula);
    
    figure(4);
    surf(T_arr, X_arr, kappa_hat_a, 'EdgeColor', 'none');
    V = axis; V(2) = t_log(end); axis(V);
    xlabel('Time [s]');
    ylabel('Space');
    zlabel('$\hat \kappa$', 'interpreter','latex');
    colormap(parula);