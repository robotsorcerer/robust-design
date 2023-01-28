% Copyright (C) Henrik Anfinsen 2017-2019
%
% Feel free to use the code, but give credit to the original source and
% author

function [K, L] = solver_2x2(params, N_g)
%% Solves a PDE system on the form
%     \mu(x) K_x - \lambda(y) K_y - a_1(x, y) K - a_2(x, y) L = 0
%     \mu(x) L_x +     \mu(y) L_y - b_1(x, y) K - b_2(x, y) L = 0
%                         K(x, x) = f(x)
%             L(x, 0) - q K(x, 0) = 0
% defined for 0 \leq \xi \leq x \leq 1,
% where 
%     lambda, mu, a_1, a_2, b_1, b_3, f 
% are known functions, and 
%     q 
% is a known constant.
% It is solved by discretizing the grid, and setting up an algebraic 
% system of equations that is solved using the backslash operator.

% % fun contains: functions
%     mu, lambda, a_1, a_2, b_1, b_2, f
% which take x as input and produces the output, and a constant
%    q.
% N_g is the number of discretization points in each direction.
% The outputs are
%     K, L
% which are N_g x N_g matrices containing the results.


Delta = 1 / (N_g - 1);

numTot = N_g*(N_g + 1) / 2; % Total number of nodes per kernel

index_map = zeros(N_g);
index_map(logical(triu(ones(N_g)) > 0)) = (1:numTot)';

dummy1 = triu(reshape(1:(N_g^2), N_g, N_g));
glob_to_mat = dummy1(dummy1 > 0);

indexes = kron([1 1 1 1 1], (1:(2*numTot))');
weights = zeros(2*numTot, 5);
RHS = zeros(2*numTot, 1);

% Directional derivatives
[s_K_x, s_K_y] = meshgrid(params.mu, -params.lambda);
[s_L_x, s_L_y] = meshgrid(params.mu, params.mu);

s_K = sqrt(s_K_x.^2 + s_K_y.^2);
s_L = sqrt(s_L_x.^2 + s_L_y.^2);

% K: Iterate over triangular domain
for indX = 1:N_g
    for indY = 1:indX
        glob_ind = index_map(indY, indX);
        if (indY == indX) % Boundary
            weights(glob_ind, 1) = 1;
            RHS(glob_ind) = params.f(indY);
        else
            theta = atan2(s_K_x(indY, indX), -s_K_y(indY, indX));
            if (theta > pi / 4) % Then it hits the left side
                sigma = Delta / sin(theta); % Vector length

                d = sigma * cos(theta); % Distance from top

                preFactor = (s_K(indY, indX) / sigma);

                % Itself
                weights(glob_ind, 1) = preFactor;

                % Top left
                weights(glob_ind, 2) = -preFactor * d / Delta;
                indexes(glob_ind, 2) = index_map(indY + 1, indX - 1);

                % Bottom left
                weights(glob_ind, 3) = -preFactor * (Delta - d) / Delta;
                indexes(glob_ind, 3) = index_map(indY, indX - 1);
            else % Then it hits the top
                sigma = Delta / cos(theta); % Vector length

                d = sigma * sin(theta); % Distance from top

                preFactor = (s_K(indY, indX) / sigma);

                % Itself
                weights(glob_ind, 1) = preFactor;

                % Top left
                weights(glob_ind, 2) = -preFactor * d / Delta;
                indexes(glob_ind, 2) = index_map(indY + 1, indX - 1);

                % Top right
                weights(glob_ind, 3) = -preFactor * (Delta - d) / Delta;
                indexes(glob_ind, 3) = index_map(indY + 1, indX);
            end;
            
            if (indY == indX - 1) % Subdiagnoal
                % Top left node DOES NOT EXIST, so instead distribute to the
                % neightbours
                if (theta > pi / 4) % Then it hits the left side
                    weights(glob_ind, 3) = weights(glob_ind, 3) + weights(glob_ind, 2);
                    weights(glob_ind, 1) = weights(glob_ind, 1) - weights(glob_ind, 2);
                    indexes(glob_ind, 2) = index_map(indY + 1, indX);
                else % Then it hits the top side
                    weights(glob_ind, 3) = weights(glob_ind, 3) + weights(glob_ind, 2);
                    weights(glob_ind, 1) = weights(glob_ind, 1) - weights(glob_ind, 2);
                    indexes(glob_ind, 2) = index_map(indY, indX - 1);
                end;
            end;
        end;
    end;
end;


% L: Iterate over triangular domain
for indX = 1:N_g
    for indY = 1:indX
        glob_ind = index_map(indY, indX) + numTot;
        if (indY == 1) % Boundary
            weights(glob_ind, 1) = - 1;
            weights(glob_ind, 2) = params.q;
            indexes(glob_ind, 2) = glob_ind - numTot;
        else
            theta = atan2(s_L_y(indY, indX), s_L_x(indY, indX));
            if (theta < pi / 4) % Then it hits the left side
                sigma = Delta / cos(theta); % Vector length

                d = sigma * sin(theta); % Distance from top

                preFactor = (s_L(indY, indX) / sigma);

                % Itself
                weights(glob_ind, 1) = preFactor;

                % Top left
                weights(glob_ind, 2) = -preFactor * (Delta - d) / Delta;
                indexes(glob_ind, 2) = index_map(indY, indX - 1) + numTot;

                % Bottom left
                weights(glob_ind, 3) = -preFactor * d / Delta;
                indexes(glob_ind, 3) = index_map(indY - 1, indX - 1) + numTot;
            else % Then it hits the bottom
                sigma = Delta / sin(theta); % Vector length

                d = sigma * cos(theta); % Distance from right

                preFactor = (s_L(indY, indX) / sigma);

                % Itself
                weights(glob_ind, 1) = preFactor;

                % Bottom left
                weights(glob_ind, 2) = -preFactor * d / Delta;
                indexes(glob_ind, 2) = index_map(indY - 1, indX - 1) + numTot;

                % Bottom right
                weights(glob_ind, 3) = -preFactor * (Delta - d) / Delta;
                indexes(glob_ind, 3) = index_map(indY - 1, indX) + numTot;
            end;
        end;
    end;
end;


% Source terms
for indX = 1:N_g
    for indY = 1:indX
        glob_ind = index_map(indY, indX);
        if (indY == indX) % Boundary
            % Do nothing
        else
            weights(glob_ind, 4) = -params.a_1(indY, indX);
            
            weights(glob_ind, 5) = -params.a_2(indY, indX);
            indexes(glob_ind, 5) = glob_ind + numTot;
        end;
        
        if (indY == 1) % Boundary
            % Do nothing
        else
            weights(glob_ind + numTot, 4) = -params.b_1(indY, indX);
            indexes(glob_ind + numTot, 4) = glob_ind;
            
            weights(glob_ind + numTot, 5) = -params.b_2(indY, indX);
        end;
    end;
end;

%% Matrix solving
counter = 0;
A_coord_x = zeros(2*numTot * 5, 1);
A_coord_y = zeros(2*numTot * 5, 1);
A_vals = zeros(2*numTot * 5, 1);

b_coord_y = zeros(2*numTot, 1);
b_vals = zeros(2*numTot, 1);

for i = 1:(2*numTot)
    for j = 1:5
        counter = counter + 1;
        A_coord_y(counter) = i;
        A_coord_x(counter) = indexes(i, j);
        A_vals(counter) = weights(i, j);
    end;
    b_coord_y(i) = i;
    b_vals(i) = RHS(i);
end;

A = sparse(A_coord_y, A_coord_x, A_vals, 2*numTot, 2*numTot);
b = sparse(b_coord_y, ones(2*numTot, 1), b_vals, 2*numTot, 1);

x = A \ b;

K = zeros(N_g);
L = zeros(N_g);

K_dummy = x(1:numTot);
L_dummy = x((numTot+1):(2*numTot));

K(glob_to_mat) = K_dummy;
L(glob_to_mat) = L_dummy;