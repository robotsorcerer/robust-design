function T_xx_vecv = Trans_kron_vecv(xn)
    
    T_xx_vecv = [];

    for i = 1: xn
        for j = i: xn
            p = zeros(1,xn^2); 
            p((i-1)*xn+j) = 1;
            T_xx_vecv = [T_xx_vecv; p];
        end
    end
end