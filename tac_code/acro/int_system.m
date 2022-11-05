function [X, U] = int_system(dt, tf, x0)
   
   N = floor(tf/dt);
   X = zeros(N+1,4);
   U = zeros(N,1);
   
   X(1,:) = x0';
   K_init = [-227.661,-90.340,-98.11,-46.112];
   
   y = zeros(1,1);
   
   for i = 1:N
       t = i*dt;
       y = y - y*dt + normrnd(0,1,[1,1])*sqrt(dt);
       u = -K_init*X(i,:)' + 10*y;
       x_next = system_pendulum(dt, X(i,:)', u);
       X(i+1,:) = x_next';
       U(i,:) = u';
   end    

end