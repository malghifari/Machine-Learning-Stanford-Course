function t = transformRes(y, K)

t = zeros(size(y,1), K);

for i = 1:size(y,1),
    t(i, y(i)) = 1;
end

end