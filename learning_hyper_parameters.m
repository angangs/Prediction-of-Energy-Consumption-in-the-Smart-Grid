function [alpha,beta] = learning_hyper_parameters(X,Y,phi_basis)
%% read data
e = 10^(-8);

%% arbitrary values to alpha and beta
alpha = rand();
beta = rand();
m = rand();

%% design matrix
phi = phi_basis(X);
covariance = inv(alpha * eye(size(phi,2)) + beta * phi' * phi );

%% log likelihood
ll = sum( log( normpdf( Y - X.*m,0,sqrt(1/beta) ) ) );
% fprintf('log likelihood = %f, alpha = %f, beta = %f\n\n', ll, alpha, beta);

%% iterate until convergence
while (true)
    %%  eigenvalues
    lamda = eig( beta * phi' * phi );
    
    %% step I given alpha and beta compute gama and mN
    mN = beta * covariance * phi' * Y;
    gama = sum(lamda./(alpha+lamda));

    %% step II given gama and mN compute alpha and beta
    alpha = gama/(mN'*mN);
    SUM=0;
    for i=1:length(X)
        SUM=SUM+(Y(i,1)-mN'*phi_basis(X(i))')^2;
    end
    beta = (size(X,1) - gama) / SUM;
    
    %% log-likelihood
    covariance = inv(alpha * eye(size(phi,2)) + beta * phi' * phi );
    ll_new = (size(phi,1)/2) * log(alpha) + (size(phi,2)/2) * log(beta) - (beta/2) * norm(Y-phi*mN)^2 + (alpha/2) * mN'*mN - (1/2) * log( det(covariance) ) - (size(phi,2)/2)*log(2*pi);
    
     %% convergence criterion
    if (abs(ll - ll_new) <= e)
        break;
    end
    
    %% update log-likelihood, theta_old := theta_new
    ll=ll_new;
%     fprintf('log likelihood = %f, alpha = %f, beta = %f\n\n', ll, alpha, beta);
end

end
