% Copyright (C) Henrik Anfinsen 2017-2019
%
% Feel free to use the code, but give credit to the original source and
% author

function [K, L, M] = solver_n1(params, N_g)


%% Solves a PDE system on the form
%     \eps_1(x) K_x - \lambda_1(y) K_y - a_1(x, y) K - a_2(x, y) L - a_3(x, y) M = 0
%     \eps_2(x) L_x - \lambda_2(y) L_y - b_1(x, y) K - b_2(x, y) L - b_3(x, y) M = 0
%        \mu(x) M_x +       \mu(y) M_y - c_1(x, y) K - c_2(x, y) L - c_3(x, y) M = 0
%                               K(x, x) = f_1(x)
%                               L(x, x) = f_2(x)
%   M(x, 0) - q_1 K(x, 0) - q_2 L(x, 0) = 0
% defined for 0 \leq \xi \leq x \leq 1,
% where 
%     mu, lambda_1, lambda_2, a_1, a_2, a_3, b_1, b_2, b_3, c_1, c_2, c_3,
%     f_1, f_2
% are known functions, and 
%     q 
% is a known constant.
% It is solved by discretizing the grid, and setting up an algebraic 
% system of equations that is solved using the backslash operator.

% % fun contains: functions
%     eps_1, eps_2, lambda_1, lambda_2, mu, 
%     a_1, a_2, a_3, b_1, b_2, b_3, c_1, c_2, c_3,
%     f_1, f_2
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

indexes = kron(ones(1, 6), (1:(3*numTot))');
weights = zeros(3*numTot, 6);
RHS = zeros(3*numTot, 1);

% Directional derivatives
[s_K_x, s_K_y] = meshgrid(params.eps_1, -params.lambda_1);
[s_L_x, s_L_y] = meshgrid(params.eps_2, -params.lambda_2);
[s_M_x, s_M_y] = meshgrid(params.mu, params.mu);

s_K = sqrt(s_K_x.^2 + s_K_y.^2);
s_L = sqrt(s_L_x.^2 + s_L_y.^2);
s_M = sqrt(s_M_x.^2 + s_M_y.^2);

% K: Iterate over triangular domain
for indX = 1:N_g
    for indY = 1:indX
        glob_ind = index_map(indY, indX);
        if (indY == indX) % Boundary
            weights(glob_ind, 1) = 1;
            RHS(glob_ind) = params.f_1(indY);
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

            if (indY == indX - 1) % Subdiagonal
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
        if (indY == indX) % Boundary
            weights(glob_ind, 1) = 1;
            RHS(glob_ind) = params.f_2(indY);
        else
            theta = atan2(s_L_x(indY, indX), -s_L_y(indY, indX));
            if (theta > pi / 4) % Then it hits the left side
                sigma = Delta / sin(theta); % Vector length

                d = sigma * cos(theta); % Distance from top

                preFactor = (s_L(indY, indX) / sigma);

                % Itself
                weights(glob_ind, 1) = preFactor;

                % Top left
                weights(glob_ind, 2) = -preFactor * d / Delta;
                indexes(glob_ind, 2) = index_map(indY + 1, indX - 1) + numTot;

                % Bottom left
                weights(glob_ind, 3) = -preFactor * (Delta - d) / Delta;
                indexes(glob_ind, 3) = index_map(indY, indX - 1) + numTot;
            else % Then it hits the top
                sigma = Delta / cos(theta); % Vector length

                d = sigma * sin(theta); % Distance from top

                preFactor = (s_L(indY, indX) / sigma);

                % Itself
                weights(glob_ind, 1) = preFactor;

                % Top left
                weights(glob_ind, 2) = -preFactor * d / Delta;
                indexes(glob_ind, 2) = index_map(indY + 1, indX - 1) + numTot;

                % Top right
                weights(glob_ind, 3) = -preFactor * (Delta - d) / Delta;
                indexes(glob_ind, 3) = index_map(indY + 1, indX) + numTot;
            end;

            if (indY == indX - 1) % Subdiagonal
                % Top left node DOES NOT EXIST, so instead distribute to the
                % neightbours
                if (theta > pi / 4) % Then it hits the left side
                    weights(glob_ind, 3) = weights(glob_ind, 3) + weights(glob_ind, 2);
                    weights(glob_ind, 1) = weights(glob_ind, 1) - weights(glob_ind, 2);
                    indexes(glob_ind, 2) = index_map(indY + 1, indX) + numTot;
                else % Then it hits the top side
                    weights(glob_ind, 3) = weights(glob_ind, 3) + weights(glob_ind, 2);
                    weights(glob_ind, 1) = weights(glob_ind, 1) - weights(glob_ind, 2);
                    indexes(glob_ind, 2) = index_map(indY, indX - 1) + numTot;
                end;
            end;
        end;
    end;
end;



% M: Iterate over triangular domain
for indX = 1:N_g
    for indY = 1:indX
        glob_ind = index_map(indY, indX) + 2*numTot;
        if (indY == 1) % Boundary
            weights(glob_ind, 1) = - 1;

            weights(glob_ind, 2) = params.q_1;
            indexes(glob_ind, 2) = glob_ind - 2*numTot;

            weights(glob_ind, 3) = params.q_2;
            indexes(glob_ind, 3) = glob_ind - numTot;
        else
            theta = atan2(s_M_y(indY, indX), s_M_x(indY, indX));
            if (theta < pi / 4) % Then it hits the left side
                sigma = Delta / cos(theta); % Vector length

                d = sigma * sin(theta); % Distance from top

                preFactor = (s_M(indY, indX) / sigma);

                % Itself
                weights(glob_ind, 1) = preFactor;

                % Top left
                weights(glob_ind, 2) = -preFactor * (Delta - d) / Delta;
                indexes(glob_ind, 2) = index_map(indY, indX - 1) + 2*numTot;

                % Bottom left
                weights(glob_ind, 3) = -preFactor * d / Delta;
                indexes(glob_ind, 3) = index_map(indY - 1, indX - 1) + 2*numTot;
            else % Then it hits the bottom
                sigma = Delta / sin(theta); % Vector length

                d = sigma * cos(theta); % Distance from right

                preFactor = (s_M(indY, indX) / sigma);

                % Itself
                weights(glob_ind, 1) = preFactor;

                % Bottom left
                weights(glob_ind, 2) = -preFactor * d / Delta;
                indexes(glob_ind, 2) = index_map(indY - 1, indX - 1) + 2*numTot;

                % Bottom right
                weights(glob_ind, 3) = -preFactor * (Delta - d) / Delta;
                indexes(glob_ind, 3) = index_map(indY - 1, indX) + 2*numTot;
            end;
        end;
    end;
end;


% Source terms
for indX = 1:N_g
    for indY = 1:indX
        glob_ind = index_map(indY, indX);
        % For K
        if (indY == indX) % Boundary
            % Do nothing
        else
            weights(glob_ind, 4) = -params.a_1(indY, indX);

            weights(glob_ind, 5) = -params.a_2(indY, indX);
            indexes(glob_ind, 5) = glob_ind + numTot;

            weights(glob_ind, 6) = -params.a_3(indY, indX);
            indexes(glob_ind, 6) = glob_ind + 2*numTot;
        end;

        % For L
        if (indY == indX) % Boundary
            % Do nothing
        else
            weights(glob_ind + numTot, 4) = -params.b_1(indY, indX);
            indexes(glob_ind + numTot, 4) = glob_ind;

            weights(glob_ind + numTot, 5) = -params.b_2(indY, indX);

            weights(glob_ind + numTot, 6) = -params.b_3(indY, indX);
            indexes(glob_ind + numTot, 6) = glob_ind + 2*numTot;
        end;

        % For M
        if (indY == 1) % Boundary
            % Do nothing
        else
            weights(glob_ind + 2*numTot, 4) = -params.c_1(indY, indX);
            indexes(glob_ind + 2*numTot, 4) = glob_ind;

            weights(glob_ind + 2*numTot, 5) = -params.c_2(indY, indX);
            indexes(glob_ind + 2*numTot, 5) = glob_ind + numTot;

            weights(glob_ind + 2*numTot, 6) = -params.c_3(indY, indX);
        end;
    end;
end;

%% Matrix solving
counter = 0;
A_coord_x = zeros(3*numTot * 6, 1);
A_coord_y = zeros(3*numTot * 6, 1);
A_vals = zeros(3*numTot * 6, 1);

b_coord_y = zeros(3*numTot, 1);
b_vals = zeros(3*numTot, 1);

for i = 1:(3*numTot)
    for j = 1:6
        counter = counter + 1;
        A_coord_y(counter) = i;
        A_coord_x(counter) = indexes(i, j);
        A_vals(counter) = weights(i, j);
    end;
    b_coord_y(i) = i;
    b_vals(i) = RHS(i);
end;

A = sparse(A_coord_y, A_coord_x, A_vals, 3*numTot, 3*numTot);
b = sparse(b_coord_y, ones(3*numTot, 1), b_vals, 3*numTot, 1);

x = A \ b;

K = zeros(N_g);
L = zeros(N_g);
M = zeros(N_g);

K_dummy = x(1:numTot);
L_dummy = x((numTot+1):(2*numTot));
M_dummy = x((2*numTot+1):(3*numTot));

K(glob_to_mat) = K_dummy;
L(glob_to_mat) = L_dummy;
M(glob_to_mat) = M_dummy;