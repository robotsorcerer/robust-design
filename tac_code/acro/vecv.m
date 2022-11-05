function xv = vecv(x)
    xv = kron(x,x);
    ij = [];
    for i = 2:length(x)
        ij = [ij, (i-1)*length(x)+1:(i-1)*length(x)+i-1];
    end
    xv(ij) = [];
end

