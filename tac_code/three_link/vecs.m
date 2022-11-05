function Pv = vecs(P)
   ii = [];
   ij = [];
   Pv = reshape(P',[size(P,1)*size(P,2),1]);
   for i = 1:size(P,2)
       ii = [ii, (i-1)*size(P,2)+i];
       ij = [ij, (i-1)*size(P,2)+1:(i-1)*size(P,2)+i-1];       
   end
   Pv = Pv*2;
   Pv(ii) = Pv(ii)/2;    
   Pv(ij) = [];
end