function [log_lik, grad_log_lik] = costFunction_gaus(x_train, y_train, theta_hyp , sigma, kernel)    
    m = size(x_train,1);
    n = size(x_train,1);
    Cn = zeros(m,n);
    
    for t = 1:m
        for l = 1:n
            Cn(t,l) = kernel(x_train(t,:), x_train(l,:),theta_hyp);
            gradCn_theta0(t,l) = exp(- ( (x_train(t,:)-x_train(l,:))'*(x_train(t,:)-x_train(l,:)) ) / ( 2 * ( theta_hyp(2) )^2 ) );
            gradCn_theta1(t,l) = theta_hyp(1) * ( (x_train(t,:)-x_train(l,:))'*(x_train(t,:)-x_train(l,:)) ) * (1/(theta_hyp(2))^3) * exp(-(x_train(t,:)-x_train(l,:))'*(x_train(t,:)-x_train(l,:)) / (2*(theta_hyp(2))^2));
        end
    end
    
    Cn = Cn + sigma^2 * eye( size(x_train,1) );
    
    log_lik = (1/2) * log( det(Cn) ) + (1/2) * y_train' * inv(Cn) * y_train + (size(Cn,1)/2) * log(2*pi);
    
    grad_log_lik_theta0 = (1/2) * trace(inv(Cn) * gradCn_theta0) - (1/2) * y_train' * inv(Cn) * gradCn_theta0 * inv(Cn) * y_train;
    grad_log_lik_theta1 = (1/2) * trace(inv(Cn) * gradCn_theta1) - (1/2) * y_train' * inv(Cn) * gradCn_theta1 * inv(Cn) * y_train;
    
    grad_log_lik = [grad_log_lik_theta0 grad_log_lik_theta1]';
end