
for i = 1:13
    a = s.(fields{i});
    csvwrite(strcat(fields{i}, ".csv"),a);
end