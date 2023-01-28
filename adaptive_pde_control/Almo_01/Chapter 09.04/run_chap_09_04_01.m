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


% Simulation specific
if (sys.ctrl_on == 1)
    N       = 100;
    simH    = 10;
    h       = 0.01;
    N_g     = 100;
else
    N       = 40;
    simH    = 5;
    h       = 0.01;
    N_g     = 40;
end;


N_grid  = N + 2;
sys.N   = N;
sys.N_grid = N_grid;
sys.N_g = N_g;
sys.simH = simH;
sys.Tspan = 0:h:simH;

sys.Delta = 1 / (N + 1);
sys.xspan = (0:sys.Delta:1)';
sys.xspanT = (sys.Delta:sys.Delta:(1 - sys.Delta))';

% System parameters
sys.lambda = 1;
sys.mu = sys.lambda;
sys.c_1 = -0.1;
sys.c_2 = 1;
sys.c_3 = 0.4;
sys.c_4 = 0.2;
sys.q = 4;

sys.kernel_Update_Interval = 0.1;
        
    
%% Controller gain

sys.k_U = 100;

sys.intArr = sys.Delta * [0.5; ones(N, 1); 0.5];

sys.gamma = 1;
sys.gamma1 = sys.gamma;
sys.gamma2 = sys.gamma;
sys.gamma3 = sys.gamma;
sys.gamma4 = sys.gamma;
sys.gamma5 = sys.gamma;

sys.rho = 1e-3;

u0 = sin(2 * pi * sys.xspanT);
v0 = sys.xspanT;
u_hat0 = zeros(N, 1);
v_hat0 = zeros(N, 1);
c1_hat0 = 0;
c2_hat0 = 0;
c3_hat0 = 0;
c4_hat0 = 0;
q_hat0 = 0;

U_f0 = 0;


x0 = [u0; v0; u_hat0; v_hat0; c1_hat0; c2_hat0; c3_hat0; c4_hat0; q_hat0; U_f0];

tic;
[t_log, x_log] = ode23(@(t, x) ode_chap_09_04_01(t, x, sys), sys.Tspan, x0);
toc;

%% Post-processing
numT = length(t_log);

xx = reshape(x_log(:, 1:(4*N)), numT, N, 4);
xx_a = zeros(numT, N + 2, 4);
xx_a(:, 1, :) = 2*xx(:, 1, :) - xx(:, 2, :);
xx_a(:, N+2, :) = 2*xx(:, N, :) - xx(:, N-1, :);
xx_a(:, 2:(N+1), :) = xx;

u_a = squeeze(xx_a(:, :, 1));       u = u_a(:, 2:(N+1));
v_a = squeeze(xx_a(:, :, 2));       v = v_a(:, 2:(N+1));
u_hat_a = squeeze(xx_a(:, :, 3));   u_hat = u_hat_a(:, 2:(N+1));
v_hat_a = squeeze(xx_a(:, :, 4));   v_hat = v_hat_a(:, 2:(N+1));
c1_hat = x_log(:, 4*N + 1);
c2_hat = x_log(:, 4*N + 2);
c3_hat = x_log(:, 4*N + 3);
c4_hat = x_log(:, 4*N + 4);
q_hat = x_log(:, 4*N + 5);

U_f = x_log(:, 4*N+6);


% Calculations

u_tilde = u_a - u_hat_a;
v_tilde = v_a - v_hat_a;

c1_tilde = sys.c_1 - c1_hat;
c2_tilde = sys.c_2 - c2_hat;
c3_tilde = sys.c_3 - c3_hat;
c4_tilde = sys.c_4 - c4_hat;
q_tilde = sys.q - q_hat;



%% Plotting
XSPAN = repmat(sys.xspan', numT, 1);
[X_arr, T_arr] = meshgrid(sys.xspan, t_log);


u_norm = sqrt((u_a.^2) * sys.intArr);
v_norm = sqrt((v_a.^2) * sys.intArr);
uv_norm = u_norm + v_norm;


    figure(1);
    plot(t_log, 0*t_log, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, uv_norm, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$||u|| + ||v||$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = uv_norm;
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;


    
    %% ACTUATION
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


%% Parameter estimates
    figure(3);
    plot(t_log, 0*t_log + sys.c_1, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, c1_hat, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$\hat c_{11}$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = [c1_hat; sys.c_1];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    

    figure(4);
    plot(t_log, 0*t_log + sys.c_2, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, c2_hat, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$\hat c_{12}$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = [c2_hat; sys.c_2];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    
    
    figure(5);
    plot(t_log, 0*t_log + sys.c_3, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, c3_hat, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$\hat c_{21}$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = [c3_hat; sys.c_3];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;

    
    
    figure(6);
    plot(t_log, 0*t_log + sys.c_4, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, c4_hat, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$\hat c_{22}$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = [c4_hat; sys.c_4];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;

    
    
    figure(7);
    plot(t_log, 0*t_log + sys.q, 'k', 'LineWidth', 2);
    hold on;
    plot(t_log, q_hat, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$\hat q$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = [q_hat; sys.q];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;