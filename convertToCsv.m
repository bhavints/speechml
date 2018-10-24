fns = fieldnames(s);
for i = 1:14
    a = s.(fns{i});
    csvwrite(strcat(fns{i}, ".csv"),a);
end