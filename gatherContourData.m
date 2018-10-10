files = dir('*.mat');
s = struct();
for k = 2:length(files)
    [p, f] = fileparts(files(k).name);
    contourData = get_tvs_from_trackfile(files(k).name);
    sequencelength = size(contourData.tv{1}.cd, 1);
    xi = 1:0.8328:sequencelength; 
    newsequencelength = size(xi, 2);
    s.(f) = zeros(newsequencelength, 6);
    mycell = [ contourData.tv{:} ];
    y = [ mycell(:).cd ];
    for i=1:6
        individualColumn=y(:,i);
        x = 1:sequencelength; 
        s.(f)(:,i) = transpose(interp1(x,individualColumn,xi));    
    end
    % Do some stuff
end