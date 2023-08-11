function [K] = compute_kernel_matrix(k, X, Z,theta)
    m = size(X,1);
    n = size(Z,1);
    K = zeros(m,n);
    for t = 1:m
        for l = 1:n
            K(t,l) = k(X(t,:), Z(l,:),theta);
        end
    end
end