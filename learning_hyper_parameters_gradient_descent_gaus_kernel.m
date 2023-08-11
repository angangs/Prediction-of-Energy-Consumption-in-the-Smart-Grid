function [theta] = learning_hyper_parameters_gradient_descent_gaus_kernel(x_train, y_train, initial_theta, sigma, kernel)
    % Tolerance
    epsilon = 3e-01;

    % Backtracking parameters
    alpha = 0.1;
    beta = 0.5;

    % Initialization
    theta_grad = initial_theta';

    [log_lik(1), grad_log_lik] = costFunction_gaus(x_train, y_train, theta_grad(:,1) , sigma, kernel);  

    Dtheta = -grad_log_lik;    

    k = 1;

    while ( norm(Dtheta(:,k)) > epsilon )
        t = 1;

        %% Check feasibility - !!! backtrack if (x + t * Dx) is not feasible !!!    
        while ~feasibility_test(x_train, y_train, theta_grad(:,k) + t * Dtheta(:,k) , sigma, kernel)
%             fprintf('\n Gradient infeasible');
            t = t * beta;
        end

        %% Backtracking line search
        [log_lik_step, grad_log_lik_step] = costFunction_gaus(x_train, y_train, theta_grad(:,k) + t * Dtheta(:,k) , sigma, kernel);  
        while(log_lik_step > log_lik(k) + alpha * t * grad_log_lik' * Dtheta(:,k))
            t = t * beta;
            [log_lik_step, grad_log_lik_step] = costFunction_gaus(x_train, y_train, theta_grad(:,k) + t * Dtheta(:,k) , sigma, kernel);
        end

        %% update step
        theta_grad(:,k+1) = theta_grad(:,k) + t * Dtheta(:,k);

        %% evaluate cost function and gradient
        [log_lik(k+1), grad_log_lik] = costFunction_gaus(x_train, y_train, theta_grad(:,k+1), sigma, kernel);    
        k = k+1;
        %% Compute new gradient direction
        Dtheta(:,k) = -grad_log_lik;
    end
    theta =theta_grad(:,k);
end