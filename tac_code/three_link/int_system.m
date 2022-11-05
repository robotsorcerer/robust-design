function [X, U] = int_system(dt, tf, x0)
   
   N = floor(tf/dt);
   X = zeros(N+1,6);
   U = zeros(N,2);
   
   X(1,:) = x0';
   K_init = [ -257.5  -96.0  -39.4  -85.6  -37.5  -19.8
 -628.4 -236.7  -94.1 -208.6  -93.3  -46.7];
   
   y = zeros(2,1);
   
   for i = 1:N
       t = i*dt;
       y = y - y*dt + normrnd(0,1,[2,1])*sqrt(dt);
       u = -K_init*X(i,:)' + 10*y;
       x_next = system_pendulum(dt, X(i,:)', u);
       X(i+1,:) = x_next';
       U(i,:) = u';
   end    

end