function A = one_hot_matrix(y)
    A = zeros(length(y),10);
    for i=1:length(y)
        vec = zeros(10,1);
        num = y(i);
        vec(num+1) = 1;
        A(i,:) = vec;
    end
end