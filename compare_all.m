clear all;close all;clc;
fprintf('Preditction process started...\n\n');

%% define train - test - target set
days=1;
N=48*days;
X=load('Data.mat');

x_input = (X.ausdata(7,1:N))';
max_x_input = max(x_input);
x_input = x_input/(max(x_input));

target = X.ausdata(8,1:N)';
max_target = max(target);
target = target/(max(target));

w = [x_input target];
w = sortrows(w);
x_input =  w(:,1);
target =  w(:,2);

cnt_time=0;
train_size = 0.8; % percent

for iter=1:10
cnt_time=cnt_time+1;
%% CASE I polynomial
num_of_thetas=3;
linear_kernel = @(x,z,theta) theta(1)+theta(2)*(x'*z).^1+theta(3)*(x'*z).^2;

%% noise level
sigma = 0.1;

%% sample gaussian process prior
initial_theta = rand(1,num_of_thetas);
initial_theta =initial_theta+5;

%% true function - training set - cross validation set
train_size = 0.8; % percent

% create training set
indices = randperm(N);
indices_train = indices(1:floor(train_size*N));
indices_test = indices(floor(train_size*N)+1:end);

if ~ismember(max(indices),indices_train)
    [flag,pos]=ismember(max(indices),indices_test);
    tmp = indices_test(pos);
    swap_index=1+floor(size(indices_train,2)*rand());
    indices_test(pos) = indices_train(swap_index);
    indices_train(swap_index)=tmp;
end

x_train = x_input(indices_train);
y_train = target(indices_train);
x_test = x_input(indices_test);
y_test = target(indices_test); 

[x_new,x_new_index] = sortrows([x_train;x_test]);

figure
plot(x_input*max_x_input,target*max_target,'b','LineWidth', 1);hold on;
plot(x_train*max_x_input,y_train*max_target,'ro');
plot(x_test*max_x_input,y_test*max_target,'kx');

%% maximize log likelihood to define hyperparameters
[theta] = learning_hyper_parameters_gradient_descent(x_train, y_train, initial_theta, sigma, linear_kernel);

%% run Gaussian process regression
cover_fill = @(x,lower,upper,color) set(fill([x;x(end:-1:1,:)],[lower;upper(end:-1:1,:)],color),'EdgeColor',color,'facealpha',0.125);

K_train_train = compute_kernel_matrix(linear_kernel,x_train,x_train,theta);
K_train_test = compute_kernel_matrix(linear_kernel,x_train,x_new,theta);
K_test_test = compute_kernel_matrix(linear_kernel,x_new,x_new,theta);

C = K_train_train + sigma^2 * eye(size(x_train,1));
mu_test = K_train_test' * (C)^(-1) * y_train;
sigma_test = K_test_test + sigma^2 * eye(size(x_new,1)) - K_train_test' * (C)^(-1) * K_train_test;

lower_test = mu_test - 2*sqrt(diag(sigma_test));
upper_test = mu_test + 2*sqrt(diag(sigma_test));

cover_fill(x_new*max_x_input,lower_test*max_target,upper_test*max_target,'b');
plot(x_new*max_x_input, mu_test*max_target, 'r', 'LineWidth', 2);

%% MSE
total_sum_test = 0;
total_sum_train = 0;

for i=1:size(x_new_index,1)
    if x_new_index(i)<=size(x_train,1)
        total_sum_train = total_sum_train + ( y_train( x_new_index(i) ) - mu_test(i) )^2;
    else
        total_sum_test = total_sum_test + ( y_test( x_new_index(i) - size(x_train,1) ) - mu_test(i) )^2;
    end
end

MSEtrain_gaus_pol(cnt_time) = total_sum_train / size(y_train,1);
MSEtest_gaus_pol(cnt_time) = total_sum_test / size(y_test,1);

avg_mean_square_error_train = sum(MSEtrain_gaus_pol)/length(MSEtrain_gaus_pol);
% fprintf('Mean square error of train set is: %f\n\n',avg_mean_square_error_train);

avg_mean_square_error_test = sum(MSEtest_gaus_pol)/length(MSEtest_gaus_pol);
fprintf('Mean square error of test set is: %f\n\n',avg_mean_square_error_test);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% CASE II gaussian kernel
num_of_thetas=2;
gaus_kernel = @(x,z,theta) theta(1)*exp(-(x-z)'*(x-z) / (2*(theta(2))^2));

%% noise level
sigma = 0.1;

%% sample gaussian process prior
initial_theta = rand(1,num_of_thetas);
initial_theta =initial_theta+5;

%% true function - training set - cross validation set

train_size = 0.8; % percent

% create training set
indices = randperm(N);
indices_train = indices(1:floor(train_size*N));
indices_test = indices(floor(train_size*N)+1:end);

if ~ismember(max(indices),indices_train)
    [flag,pos]=ismember(max(indices),indices_test);
    tmp = indices_test(pos);
    swap_index=1+floor(size(indices_train,2)*rand());
    indices_test(pos) = indices_train(swap_index);
    indices_train(swap_index)=tmp;
end

x_train = x_input(indices_train);
y_train = target(indices_train);
x_test = x_input(indices_test);
y_test = target(indices_test); 

[x_new2,x_new_index] = sortrows([x_train;x_test]);

%% maximize log likelihood to define hyperparameters
[theta] = learning_hyper_parameters_gradient_descent_gaus_kernel(x_train, y_train, initial_theta, sigma, gaus_kernel);

%% run Gaussian process regression
K_train_train = compute_kernel_matrix(gaus_kernel,x_train,x_train,theta);
K_train_test = compute_kernel_matrix(gaus_kernel,x_train,x_new2,theta);
K_test_test = compute_kernel_matrix(gaus_kernel,x_new2,x_new2,theta);

C = K_train_train + sigma^2 * eye(size(x_train,1));
mu_test2 = K_train_test' * (C)^(-1) * y_train;
sigma_test = K_test_test + sigma^2 * eye(size(x_new2,1)) - K_train_test' * (C)^(-1) * K_train_test;

lower_test2 = mu_test2 - 2*sqrt(diag(sigma_test));
upper_test2 = mu_test2 + 2*sqrt(diag(sigma_test));
cover_fill(x_new2*max_x_input,lower_test2*max_target,upper_test2*max_target,'g');

plot(x_new2*max_x_input, mu_test2*max_target, 'k', 'LineWidth', 2);
legend('(X,Y)','(X_{train},Y_{train})', '(X_{test},Y_{test})','variance of GP polyn.','mean of GP polyn.','variance of GP gaus.','mean of GP gaus.');
% axis([min(x_input*max_x_input) max(x_input*max_x_input) min(target*max_target) max(target*max_target)]);
xlabel('price per kWh')
ylabel('kWh')
hold off

figure
plot(x_input*max_x_input,target*max_target,'b','LineWidth', 1);hold on;
plot(x_train*max_x_input,y_train*max_target,'ro');
plot(x_test*max_x_input,y_test*max_target,'kx');

cover_fill(x_new*max_x_input,lower_test*max_target,upper_test*max_target,'b');
plot(x_new*max_x_input, mu_test*max_target, 'r', 'LineWidth', 2);

cover_fill(x_new2*max_x_input,lower_test2*max_target,upper_test2*max_target,'g');
plot(x_new2*max_x_input, mu_test2*max_target, 'k', 'LineWidth', 2);

%% MSE
total_sum_test = 0;
total_sum_train = 0;

for i=1:size(x_new_index,1)
    if x_new_index(i)<=size(x_train,1)
        total_sum_train = total_sum_train + ( y_train( x_new_index(i) ) - mu_test2(i) )^2;
    else
        total_sum_test = total_sum_test + ( y_test( x_new_index(i) - size(x_train,1) ) - mu_test2(i) )^2;
    end
end

MSEtrain_gaus_gaus(cnt_time) = total_sum_train / size(y_train,1);
MSEtest_gaus_gaus(cnt_time) = total_sum_test / size(y_test,1);

avg_mean_square_error_train = sum(MSEtrain_gaus_gaus)/length(MSEtrain_gaus_gaus);
% fprintf('Mean square error of train set is: %f\n\n',avg_mean_square_error_train);
avg_mean_square_error_test = sum(MSEtest_gaus_gaus)/length(MSEtest_gaus_gaus);
fprintf('Mean square error of test set is: %f\n\n',avg_mean_square_error_test);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% BAYESIAN LINEAR REGRESSION
phi_basis = @(x) [x.^0 x.^1 x.^2 x.^3];
num_of_basis = 4;

%% DESIGN MATRIX (POLYNOMIAL BASIS)
phi = phi_basis(x_train);

%% Alpha, Beta Parameters for Polynomial
[alpha,beta] = learning_hyper_parameters(x_train,y_train,phi_basis);

%% COVARIANCE -- MEAN polynomial
covariance = inv((alpha * eye(size(phi,2)) + beta * phi' * phi ));
mean = beta * covariance * phi' * y_train;

%% plots showing the true function, the data points, and the mean and variance of predictive distribution POLYNOMIAL
cover_fill = @(x,lower,upper,color) set(fill([x;x(end:-1:1,:)],[lower;upper(end:-1:1,:)],color),'EdgeColor',color,'facealpha',0.125);

[x_total,x_new_index] = sortrows([x_train;x_test]);

for i=1:size(x_total,1)
    lower_test(i,1) = mean' * phi_basis( x_total(i) )' - 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
    upper_test(i,1) = mean' * phi_basis( x_total(i) )' + 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
end
cover_fill(x_total*max_x_input,lower_test*max_target,upper_test*max_target,'m');

pred = mean' * phi_basis(x_total)';
plot(x_total*max_x_input,pred*max_target,'c', 'LineWidth', 2);
legend('(X,Y)','(X_{train},Y_{train})', '(X_{test},Y_{test})','variance of GP polyn.','mean of GP polyn.','variance of GP gaus.','mean of GP gaus.','variance of bayesian linear regression','mean of bayesian linear regression');
% axis([min(x_input*max_x_input) max(x_input*max_x_input) min(target*max_target) max(target*max_target)]);
xlabel('price per kWh')
ylabel('kWh')
hold off

%% MSE
total_sum_test = 0;
total_sum_train = 0;
total_sum_cv = 0;


for i=1:size(x_new_index,1)
    if x_new_index(i)<=size(x_train,1)
        total_sum_train = total_sum_train + ( y_train( x_new_index(i) ) - pred(i) )^2;
    else
        total_sum_test = total_sum_test + ( y_test( x_new_index(i) - ( size(x_train,1)  ) ) - pred(i) )^2;
    end
end

MSEtrain_bayes(cnt_time) = total_sum_train / size(y_train,1);

MSEtest_bayes(cnt_time) = total_sum_test / size(y_test,1);

avg_mean_square_error_train = sum(MSEtrain_bayes)/length(MSEtrain_bayes);
avg_mean_square_error_test = sum(MSEtest_bayes)/length(MSEtest_bayes);
% fprintf('Mean square error of train set is: %f\n\n',avg_mean_square_error_train);
fprintf('Mean square error of test set is: %f\n\n',avg_mean_square_error_test);
end