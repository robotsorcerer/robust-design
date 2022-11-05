function T_vt = Trans_vec(nr,nc)
%Calculate the transformation matrix from vec(X) to vec(X')
%   X has nr rows and nc columns
%   vec(X') = T_vt*vec(X)
    
    T_vt = [];

    for i = 1: nr
        for j = 1: nc
            p = zeros(1,nr*nc); 
            p((j-1)*nr+i) = 1;
            T_vt = [T_vt; p];
        end
    end
    
end