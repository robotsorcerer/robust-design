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

startTime = tic;

if (sys.ctrl_on == 1)
    N = 80;
    sys.simHoriz = 15;
    sys.h = 0.01;
    sys.N_g = 20;
else
    N = 10;
    sys.simHoriz = 10;
    sys.h = 0.05;
    sys.N_g = 40;
end;

sys.kernel_Update_Interval = 0.1;

sys.N         = N;
sys.numChar   = N;
sys.Tspan     = 0:sys.h:sys.simHoriz;
sys.Delta     = 1 / (sys.N - 1);
sys.xspan     = linspace(0, 1, sys.N)';

sys.lambda    = 1;
sys.mu        = 2;

sys.c1        = sys.xspan .* sin(sys.xspan) + 1;
sys.c2        = cosh(sys.xspan);

sys.theta     = sys.c1 / sys.lambda;
sys.kappa     = sys.c2 / sys.mu;

sys.q   = 2;

% sys.U_func = @(t)(sin(3*t) + sin(3*0.5 * sqrt(3) * t) + sin(2*t) + sin(0.5 * pi * t));
sys.U_func = @(t)(0);

sys.gamma0 = 1;
sys.gamma1 = 1;
sys.gamma2 = 1;

sys.gamma_BC = 500;
sys.gamma = 1000;

%% Setup
[u_sti_w, u_sti_i, v_sti_w, v_sti_i] = setup_chap_09_04_02(sys);
sys.u_sti_w = u_sti_w;
sys.u_sti_i = u_sti_i;
sys.v_sti_w = v_sti_w;
sys.v_sti_i = v_sti_i;
sys.u_i = (2:N)';
sys.v_i = (1:(N-1))';

sys.intArr = sys.Delta * [0.5; ones(N-2, 1); 0.5];


%% Initial conditions
% uv_0 = zeros(3*N, 1);
u_0 = sin(sys.xspan);
v_0 = cosh(sys.xspan) .* cos(2*pi * sys.xspan);
eta_0 = zeros(N, 1);
phi_0 = zeros(N, 1);
theta_hat_0 = zeros(N, 1);
kappa_hat_0 = zeros(N, 1);
M_filt_0 = zeros(N^2, 1);
N_filt_0 = zeros(N^2, 1);
q_hat_0 = 0;


x0 = [u_0; v_0; eta_0; phi_0; theta_hat_0; kappa_hat_0; M_filt_0; N_filt_0; q_hat_0];

%% Simulate
disp('Simulating');
tic;
[t_log, x_log] = ode23(@(t,x) ode_chap_09_04_02(t, x, sys), sys.Tspan, x0);
toc;

totalTime = toc(startTime);
display(totalTime);

%% States extraction
numT = length(t_log);

states_dummy = reshape(x_log(:, 1:(6*N)), numT, N, 6);

u         = states_dummy(:, :, 1);
v         = states_dummy(:, :, 2);
eta       = states_dummy(:, :, 3);
phi       = states_dummy(:, :, 4);
theta_hat = states_dummy(:, :, 5);
kappa_hat = states_dummy(:, :, 6);


dummy_states2 = reshape(x_log(:, (6*sys.N+1):(6*sys.N+2*N^2)), numT, N, 2*N);
M_filt  = dummy_states2(:, :, 1:(N));
N_filt  = dummy_states2(:, :, (N+1):(2*N));

q_hat = x_log(:, 6*sys.N+2*N^2 + 1);


U = v(:, N);
if (sys.ctrl_on == 1)
    U(1) = 2*U(2) - U(3);
else
    U = 0*U;
end;

lambda  = sys.lambda;
mu      = sys.mu;


u_norm = sqrt((u.^2) * sys.intArr);
v_norm = sqrt((v.^2) * sys.intArr);
uv_norm = u_norm + v_norm;


    %% STATE NORMS
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
    plot(t_log, U, sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('Time [s]');
    ylabel('$U$', 'interpreter','latex');
    V = axis; V(2) = t_log(end); axis(V);
    dataBlock = U;
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;

    
%% Combined parameter plot
    figure(3);
    plot(sys.xspan, sys.theta, 'k', 'LineWidth', 2);
    hold on;
    plot(sys.xspan, theta_hat(end, :)', sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('x');
    ylabel('$\hat \theta$', 'interpreter','latex');
    V = axis; V(2) = 1; axis(V);
    dataBlock = [theta_hat(end, :)'; sys.theta];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    
    
    figure(4);
    plot(sys.xspan, sys.kappa, 'k', 'LineWidth', 2);
    hold on;
    plot(sys.xspan, kappa_hat(end, :)', sys.style3, 'Color', sys.RGB_color1, 'LineWidth', 2);
    hold off;
    xlabel('x');
    ylabel('$\hat \kappa$', 'interpreter','latex');
    V = axis; V(2) = 1; axis(V);
    dataBlock = [kappa_hat(end, :)'; sys.kappa];
    d = max(dataBlock) - min(dataBlock);
    if (d > 0)
        V(3) = min(dataBlock) - 0.2*d;
        V(4) = max(dataBlock) + 0.2*d;
        axis(V);
    end;
    
    
    figure(5);
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