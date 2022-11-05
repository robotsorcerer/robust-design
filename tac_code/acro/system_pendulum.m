function x_next = system_pendulum(dt, x, u)
    A = [zeros(2), eye(2);
        12.49 -12.54 0 0;
        -14.49 29.36 0 0];

    B = [0; 0; -2.98; 5.98];

    D = [zeros(2,2); 1*eye(2)];
    
    xi = normrnd(0,1,[2,1]);
    
    x_next = x + (A*x + B*u)*dt + D*xi*sqrt(dt);
    
end