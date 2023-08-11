function [feas] = feasibility_test(x_train, y_train, theta_hyp , sigma, kernel)
    
    m = size(x_train,1);
    n = size(y_train,1);
    Cn = zeros(m,n);

    for t = 1:m
        for l = 1:n
            Cn(t,l) = kernel(x_train(t,:), x_train(l,:),theta_hyp);
        end
    end
    
    Cn = Cn + sigma^2 * eye( size(x_train,1) );
    
    feas = (det(Cn)>0);
end