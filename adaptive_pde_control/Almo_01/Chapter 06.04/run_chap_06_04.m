% Copyright (C) Henrik Anfinsen 2017-2019
%
% Feel free to use the code, but give credit to the original source and
% author

clear all;
close all;

sys.ctrl_on = 1;
sys.second_order = 0;
sys.RGB_color1 = [1 0 0];
sys.RGB_color2 = [0 0 1];
sys.RGB_color3 = [0 1 0];
sys.style1 = '-';
sys.style2 = '-.';
sys.style3 = '--';


if (sys.ctrl_on == 1)
    sys.simH    = 15;
    sys.N       = 10;
else
    sys.simH    = 5;
    sys.N       = 10;
end;

% Simulation specific
sys.N_grid  = sys.N + 2;
sys.M       = 4 * sys.N;
sys.M_grid  = sys.M + 2;
sys.h       = 0.01;
% sys.h       = sys.simH;
sys.Tspan   = 0:sys.h:sys.simH;
 
sys.Delta = 1 / (sys.N + 1);
sys.Delta_M = 1 / (sys.M + 1);
sys.xspan = (0:sys.Delta:1)';
sys.xspanT = (sys.Delta:sys.Delta:(1 - sys.Delta))';
sys.xspanM = linspace(0, 1, sys.M_grid)';

sys.intArr = sys.Delta * [0.5; ones(sys.N, 1); 0.5];

sys.rho_hat_L = 0.1;
sys.rho_hat_U = 10;
% sys.rho_hat_L = 1;
% sys.rho_hat_U = 1;

% System parameters
sys.gamma = 4;
sys.a = 1;
sys.epsilon = 0.2;

sys.b = sqrt(sys.a / sys.epsilon);

sys.one = ones(sys.N_grid, 1);
sys.mu = 0*sys.xspan + 1 / sys.epsilon;
sys.f = 0*sys.xspan;
sys.g = - sys.gamma * sys.b * sinh(sys.b * sys.xspan);
sys.h_f = sys.gamma * sys.b^2 * cosh(sys.b*(sys.xspan * sys.one' - sys.one * sys.xspan'));
% sys.h = sys.h';

sys.mu_bar = 1 / (sys.intArr' * (1 ./ sys.mu));

sys.k_1 = 2;
sys.k_2 = 2;

sys.r_func = @(t)((1 + sin(pi * t - pi / 2)) .* (t <= 10));


%% Tuning
sys.gamma1 = 20;
sys.gamma2 = 20;

sys.k_U = 100;

N = sys.N;
M = sys.M;

%% Initial conditions
u_0 = sys.xspanT;
% u_0 = zeros(N, 1);
psi_0 = zeros(N, 1);
phi_0 = zeros(N, 1);
b_0 = zeros(N, 1);
theta_hat_0 = zeros(N, 1);
M2_0 = zeros(M, 1);
rho_hat0 = 1;
r_f_0 = 0;
U_f_0 = 0;

%% Initial condition
x0 = [u_0; psi_0; phi_0; b_0; theta_hat_0; ...
        M2_0; rho_hat0; r_f_0; U_f_0;];

%% Simulate
tic;
[t_log, x_log] = ode45(@(t, x) ode_chap_06_04(t, x, sys), sys.Tspan, x0);
toc;


%% Post-processing
numT = length(t_log);

xx1 = reshape(x_log(:, 1:(5*sys.N)), numT, sys.N, 5);
xx1_a = zeros(numT, sys.N + 2, 5);
xx1_a(:, 1, :) = 2*xx1(:, 1, :) - xx1(:, 2, :);
xx1_a(:, sys.N+2, :) = 2*xx1(:, sys.N, :) - xx1(:, sys.N-1, :);
xx1_a(:, 2:(sys.N+1), :) = xx1;

u_a = squeeze(xx1_a(:, :, 1));           u = u_a(:, 2:(sys.N+1));
psi_a = squeeze(xx1_a(:, :, 2));         psi = psi_a(:, 2:(sys.N+1));
phi_a = squeeze(xx1_a(:, :, 3));         phi = phi_a(:, 2:(sys.N+1));
b_a = squeeze(xx1_a(:, :, 4));           b = b_a(:, 2:(sys.N+1));
theta_hat_a = squeeze(xx1_a(:, :, 5));   theta_hat = theta_hat_a(:, 2:(sys.N+1));

xx2 = reshape(x_log(:, (5*sys.N+1):(5*sys.N+sys.M)), numT, sys.M, 1);
xx2_a = zeros(numT, sys.M + 2, 1);
xx2_a(:, 1, :) = 2*xx2(:, 1, :) - xx2(:, 2, :);
xx2_a(:, sys.M+2, :) = 2*xx2(:, sys.M, :) - xx2(:, sys.M-1, :);
xx2_a(:, 2:(sys.M+1), :) = xx2;

M2_a = squeeze(xx2_a(:, :, 1));          M2 = M2_a(:, 2:(sys.M+1));

rho_hat = x_log(:, 5*sys.N + sys.M + 1);
r_f = x_log(:, 5*sys.N + sys.M + 2);
U_f = x_log(:, 5*sys.N + sys.M + 3);

r_f(1) = 2*r_f(2) - r_f(3);
U_f(1) = 2*U_f(2) - U_f(3);


u_hat_a = zeros(size(u_a));
for k = 1:numT
    u_hat_a(k, :) = rho_hat(k) * psi_a(k, :) - b_a(k, :);
    for i = 1:sys.N_grid
        for j = i:sys.N_grid
            u_hat_a(k, i) = u_hat_a(k, i) + sys.Delta * (1 - 0.5 * (j == i) - 0.5 * (j == sys.N_grid)) ...
                        * theta_hat_a(k, j) * phi_a(k, sys.N_grid - j + i);
        end;
    end;

    % Second integral: theta and M
    for i = 1:sys.N_grid
        x = sys.Delta * (i - 1);
        for j = 1:sys.N_grid
            xi = sys.Delta * (j - 1);

            val = 0.5 * (1 + x - xi);
            ind = round(val / sys.Delta_M) + 1;
            u_hat_a(k, i) = u_hat_a(k, i) + sys.Delta * (1 - 0.5 * (j == 1) - 0.5 * (j == sys.N_grid)) ...
                        * theta_hat_a(k, j) * M2_a(k, ind);
        end;
    end;
end;


%% Plotting
[X_arr, T_arr] = meshgrid(sys.xspan, t_log);
[X_arr2, T_arr2] = meshgrid(sys.xspanM, t_log);

u_norm = sqrt(u_a.^2 * sys.intArr);
u_hat_norm = sqrt(u_hat_a.^2 * sys.intArr);
psi_norm = sqrt(psi_a.^2 * sys.intArr);
phi_norm = sqrt(phi_a.^2 * sys.intArr);
e_of_norm = sqrt((u_a - u_hat_a).^2 * sys.intArr);



%%% SYSTEM STATES
    % STATE FEEDBACK
    figure(1);
    surf(T_arr, X_arr, u_a, 'EdgeColor', 'none')
    V = axis; V(2) = t_log(end); axis(V);
    xlabel('Time [s]');
    ylabel('Space');
    zlabel('State $u$', 'interpreter','latex');
    view([25 36]);
    colormap(parula);
    
    
    % ESTIMATION ERROR FEEDBACK
    figure(2);
    surf(T_arr, X_arr, u_a - u_hat_a, 'EdgeColor', 'none')
    V = axis; V(2) = t_log(end); axis(V);
    xlabel('Time [s]');
    ylabel('Space');
    zlabel('Error $u - \hat u$', 'interpreter','latex');
    view([25 36]);
    colormap(parula);
    
   
%%% STATES NORMS
    figure(3);
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, u_norm, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$||u||$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    d = (max(u_norm) - min(u_norm));
    if (d > 0)
        V(3) = min(u_norm) - 0.2*d;
        V(4) = max(u_norm) + 0.2*d;
        axis(V);
    end;
    
    %% ESTIMATION ERROR
    figure(4);
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, e_of_norm, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$||u - \hat u||$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    d = (max(e_of_norm) - min(e_of_norm));
    if (d > 0)
        V(3) = min(e_of_norm) - 0.2*d;
        V(4) = max(e_of_norm) + 0.2*d;
        axis(V);
    end;

    
    % COMBINED PLOT
    figure(5);
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, u_norm, sys.style1, 'Color', sys.RGB_color1, 'LineWidth', 2); % Red
    plot(t_log, psi_norm, sys.style2, 'Color', sys.RGB_color2, 'LineWidth', 2); % Blue
    plot(t_log, phi_norm, sys.style3, 'Color', sys.RGB_color3, 'LineWidth', 2); % Green
    hold off;
    xlabel('Time [s]');
    ylabel('$||u||$, $||\psi||$ and $||\phi||$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = [u_norm; psi_norm; phi_norm];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;

   
%% Actuation
    figure(6);
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
    
    
%% theta_hat
    figure(7);
    surf(T_arr, X_arr, theta_hat_a, 'EdgeColor', 'none');
    dataBlock = theta_hat_a(:);
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(1) = 0;
        V(2) = t_log(end);
        V(3) = 0;
        V(4) = 1;
        V(5) = min(dataBlock) - 0.2*d;
        V(6) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    xlabel('Time [s]');
    ylabel('Space');
    zlabel('~~$\hat \theta$~~', 'interpreter','latex');
    view([53 32]);
    colormap(parula);
    
    
    figure(8);
    plot(t_log, 0*t_log + sys.k_1*sys.k_2, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, rho_hat, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('x');
    ylabel('$\rho$ and $\hat \rho$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = [rho_hat; sys.k_1*sys.k_2];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    

    figure(9);
    plot(t_log, b_a(:, 1), 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, sys.k_2 * u_a(:, 1), sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('x');
    ylabel('$y_r$ and $y$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = sys.r_func(t_log);
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;