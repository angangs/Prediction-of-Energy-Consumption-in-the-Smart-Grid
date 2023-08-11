clear all;close all;clc;
fprintf('Preditction process started...\n\n');

%% define train - test - target set
days=1;
N=48*days;
total_iterations=3;
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

%% CASE II gaussian kernel
num_of_thetas=2;
gaus_kernel = @(x,z,theta) theta(1)*exp(-(x-z)'*(x-z) / (2*(theta(2))^2));

%% noise level
sigma = 0.1;

%% sample gaussian process prior
num_samples = 5;
prior = zeros(size(x_input,1), num_samples);
initial_theta = rand(1,num_of_thetas);
initial_theta =initial_theta+5;

fprintf('Initial thetas are: \n');
fprintf('%f\n',initial_theta);
theta = initial_theta;

for i=1:num_samples
    prior(:,i) = sample_gp_prior(gaus_kernel,x_input,initial_theta);
end

figure;
plot(repmat(x_input,1,num_samples), prior, 'LineWidth', 2);
title(sprintf('Samples from GP'));

%% true function - training set - cross validation set
y_min = min(target) - 1;
y_max = max(target) + 1;
cnt_time=1;
train_size = 0.8; % percent

for cnt_time=1:total_iterations
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
    plot(x_input*max_x_input,target*max_target,'k','LineWidth', 1);hold on;
    plot(x_train*max_x_input,y_train*max_target,'ro');
    plot(x_test*max_x_input,y_test*max_target,'kx');

    %% maximize log likelihood to define hyperparameters
    fprintf('Optimized thetas are: \n')
    [theta] = learning_hyper_parameters_gradient_descent_gaus_kernel(x_train, y_train, initial_theta, sigma, gaus_kernel);
    fprintf('%f\n\n',theta);

    %% run Gaussian process regression
    cover_fill = @(x,lower,upper,color) set(fill([x;x(end:-1:1,:)],[lower;upper(end:-1:1,:)],color),'EdgeColor',color,'facealpha',0.125);

    K_train_train = compute_kernel_matrix(gaus_kernel,x_train,x_train,theta);
    K_train_test = compute_kernel_matrix(gaus_kernel,x_train,x_new,theta);
    K_test_test = compute_kernel_matrix(gaus_kernel,x_new,x_new,theta);

    C = K_train_train + sigma^2 * eye(size(x_train,1));
    mu_test = K_train_test' * (C)^(-1) * y_train;
    sigma_test = K_test_test + sigma^2 * eye(size(x_new,1)) - K_train_test' * (C)^(-1) * K_train_test;

    lower_test = mu_test - 2*sqrt(diag(sigma_test));
    upper_test = mu_test + 2*sqrt(diag(sigma_test));
    
    cover_fill(x_new*max_x_input,lower_test*max_target,upper_test*max_target,'b');
    plot(x_new*max_x_input, mu_test*max_target, 'g', 'LineWidth', 2);
    
    legend('(X,Y)','(X_{train},Y_{train})', '(X_{test},Y_{test})','variance','mean');
    title('Gaussian process regression using gaussian kernel');
    xlabel('price per kWh')
    ylabel('kWh')
    hold off;

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

    MSEtrain(cnt_time) = total_sum_train / size(y_train,1);
%     fprintf('Mean square error training set is: %f\n\n',MSEtrain(cnt_time));

    MSEtest(cnt_time) = total_sum_test / size(y_test,1);
    fprintf('Mean square error test set is: %f\n\n',MSEtest(cnt_time));
end

avg_mean_square_error_train = sum(MSEtrain)/length(MSEtrain);
% fprintf('Average mean square error of train set is: %f\n\n',avg_mean_square_error_train);

avg_mean_square_error_test = sum(MSEtest)/length(MSEtest);
fprintf('Average mean square error of test set is: %f\n\n',avg_mean_square_error_test);
