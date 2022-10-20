function P = vecs_inv(Pv)

dim_P = int32((-1+sqrt(1+8*length(Pv)))/2);
P = zeros(dim_P);
ij = 0;

for i=1:dim_P
    for j=i:dim_P
        ij=ij+1;
        if i==j
            P(i,j)=Pv(ij);
        else            
            P(i,j)=0.5*Pv(ij);
            P(j,i)=P(i,j);
        end
    end
end

end