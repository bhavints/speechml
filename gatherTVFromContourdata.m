s = struct();
for k = 1:length(contourdata.fl)
    f = contourdata.fl(k);
    f = f{1};
    [p, f] = fileparts(f);
    indices = find(contourdata.File==k);
    sequencelength = length(indices);
    xi = 1:0.8328:sequencelength; 
    newsequencelength = size(xi, 2);
    s.(f) = zeros(newsequencelength, 6);
    mycell = [ contourdata.tv{:} ];
    y = [ mycell(:).cd ];
    for i=1:6
        individualColumn=y(indices,i);
        x = 1:sequencelength; 
        s.(f)(:,i) = transpose(interp1(x,individualColumn,xi));    
    end
end