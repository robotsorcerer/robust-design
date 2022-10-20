function x_next = system_pendulum(dt, x, u)
    A = [zeros(3), eye(3);
        12.54 -8.26 -0.39 -0.043 2.75 -0.36;
        -4.38 36.95 -3.00 0.086 -9.57 2.29;
        -6.82 -22.94 11.93 -0.034 6.82 -2.86;];

    B = [zeros(3,2);
        -50.0 6.12;
        174.4 -38.93;
        -124.2 48.62;];

    D = [zeros(3,3);1*eye(3)];
    
    xi = normrnd(0,1,[3,1]);
    
    x_next = x + (A*x + B*u)*dt + D*xi*sqrt(dt);
    
end