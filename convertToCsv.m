fns = fieldnames(s);
for i = 1:k
    a = s.(fns{i});
    csvwrite(strcat(fns{i}, ".csv"),a);
end