
sixDistancesUpsampled = zeros(2999,6);
for i=1:6
    individualColumn=uscvtsfm001rtbVtr1sixDistances(:,i);
    x = 1:2498; 
    xi = 1:0.8328:2498; 
    sixDistancesUpsampled(:,i) = transpose(interp1(x,individualColumn,xi));
end
