function [Trans_v_vs, Trans_vs_v ] = Trans_vec_vecs(xn)
    
Trans_v_vs = [];

for i = 1: xn
    for j = i: xn
        p = zeros(1,xn^2); 
        if j==i
            p((i-1)*xn+j) = 1;
        else
            p((i-1)*xn+j) = 2;
        end
        Trans_v_vs = [Trans_v_vs; p];
    end
end

Trans_vs_v = zeros(xn^2, floor((xn+1)*xn/2));
idx_end = 0;

for i = 1: xn
    for j = i: xn
        p = zeros(1, floor((xn+1)*xn/2)); 
        if j==i
            p(idx_end+(j-i+1)) = 1;
            Trans_vs_v((i-1)*xn+j,:) = p;
        else
            p(idx_end+(j-i+1)) = 0.5;
            Trans_vs_v((i-1)*xn+j,:) = p;
            Trans_vs_v((j-1)*xn+i,:) = p;
        end        
    end
    idx_end = idx_end + (xn-i+1);
end


end

