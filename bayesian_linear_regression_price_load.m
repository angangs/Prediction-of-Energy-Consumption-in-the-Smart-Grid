clear all; close all; clc;
fprintf('Preditction process started...\n\n');

%% DATA
days = 1;
N=days*48;
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

cnt_time=1;
count_prc=0;
train_size=0.8;
basis_f=8;
%% CREATE SETS
indices = randperm(N);
indices_train = indices(1:floor(0.6*N));
indices_test = indices(floor(0.6*N)+1:end);

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

%% TRAIN-TEST PLOT
figure
plot(x_input*max_x_input,target*max_target,'b','LineWidth', 1);hold on;
plot(x_train*max_x_input,y_train*max_target,'ro');
plot(x_test*max_x_input,y_test*max_target,'kx');

%% 1st degree
phi_basis = @(x) [x.^0 x.^1];
num_of_basis = 2;

%% DESIGN MATRIX (POLYNOMIAL BASIS)
phi = phi_basis(x_train);

%% Alpha, Beta Parameters for Polynomial
[alpha,beta] = learning_hyper_parameters(x_train,y_train,phi_basis);

%% COVARIANCE -- MEAN polynomial
covariance = inv((alpha * eye(size(phi,2)) + beta * phi' * phi ));
mean2 = beta * covariance * phi' * y_train;

%% plots showing the true function, the data points, and the mean and variance of predictive distribution POLYNOMIAL
cover_fill = @(x,lower,upper,color) set(fill([x;x(end:-1:1,:)],[lower;upper(end:-1:1,:)],color),'EdgeColor',color,'facealpha',0.125);

[x_total,x_new_index] = sortrows([x_train;x_test]);

for i=1:size(x_total,1)
    lower_test(i,1) = mean2' * phi_basis( x_total(i) )' - 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
    upper_test(i,1) = mean2' * phi_basis( x_total(i) )' + 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
end
pred = mean2' * phi_basis(x_total)';
% cover_fill(x_total,lower_test,upper_test,'g');
% plot(x_total,pred,'k');

%% MSE
total_sum_test = 0;
total_sum_train = 0;

for i=1:size(x_new_index,1)
    if x_new_index(i)<=size(x_train,1)
        total_sum_train = total_sum_train + ( y_train( x_new_index(i) ) - pred(i) )^2;
    else
        total_sum_test = total_sum_test + ( y_test( x_new_index(i) - ( size(x_train,1)  ) ) - pred(i) )^2;
    end
end

MSEtrain(cnt_time) = total_sum_train / size(y_train,1);
% fprintf('Mean square error training set is: %f\n\n',MSEtrain(cnt_time));

MSEtest(cnt_time) = total_sum_test / size(y_test,1);
fprintf('Mean square error test set is: %f\n\n',MSEtest(cnt_time));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 2nd degree
phi_basis = @(x) [x.^0 x.^1 x.^2];
num_of_basis = 3;

%% DESIGN MATRIX (POLYNOMIAL BASIS)
phi = phi_basis(x_train);

%% Alpha, Beta Parameters for Polynomial
[alpha,beta] = learning_hyper_parameters(x_train,y_train,phi_basis);

%% COVARIANCE -- MEAN polynomial
covariance = inv((alpha * eye(size(phi,2)) + beta * phi' * phi ));
mean2 = beta * covariance * phi' * y_train;

%% plots showing the true function, the data points, and the mean and variance of predictive distribution POLYNOMIAL
% cover_fill = @(x,lower,upper,color) set(fill([x;x(end:-1:1,:)],[lower;upper(end:-1:1,:)],color),'EdgeColor',color,'facealpha',0.125);

[x_total,x_new_index] = sortrows([x_train;x_test]);

for i=1:size(x_total,1)
    lower_test(i,1) = mean2' * phi_basis( x_total(i) )' - 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
    upper_test(i,1) = mean2' * phi_basis( x_total(i) )' + 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
end
pred = mean2' * phi_basis(x_total)';
cover_fill(x_total*max_x_input,lower_test*max_target,upper_test*max_target,'g');
plot(x_total*max_x_input,pred*max_target,'k');

%% MSE
total_sum_test = 0;
total_sum_train = 0;

for i=1:size(x_new_index,1)
    if x_new_index(i)<=size(x_train,1)
        total_sum_train = total_sum_train + ( y_train( x_new_index(i) ) - pred(i) )^2;
    else
        total_sum_test = total_sum_test + ( y_test( x_new_index(i) - ( size(x_train,1)  ) ) - pred(i) )^2;
    end
end

MSEtrain(cnt_time) = total_sum_train / size(y_train,1);
% fprintf('Mean square error training set is: %f\n\n',MSEtrain(cnt_time));

MSEtest(cnt_time) = total_sum_test / size(y_test,1);
fprintf('Mean square error test set is: %f\n\n',MSEtest(cnt_time));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 3rd degree
phi_basis = @(x) [x.^0 x.^1 x.^2 x.^3];
num_of_basis = 4;

%% DESIGN MATRIX (POLYNOMIAL BASIS)
phi = phi_basis(x_train);

%% Alpha, Beta Parameters for Polynomial
[alpha,beta] = learning_hyper_parameters(x_train,y_train,phi_basis);

%% COVARIANCE -- MEAN polynomial
covariance = inv((alpha * eye(size(phi,2)) + beta * phi' * phi ));
mean2 = beta * covariance * phi' * y_train;

%% plots showing the true function, the data points, and the mean and variance of predictive distribution POLYNOMIAL
% cover_fill = @(x,lower,upper,color) set(fill([x;x(end:-1:1,:)],[lower;upper(end:-1:1,:)],color),'EdgeColor',color,'facealpha',0.125);

[x_total,x_new_index] = sortrows([x_train;x_test]);

for i=1:size(x_total,1)
    lower_test(i,1) = mean2' * phi_basis( x_total(i) )' - 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
    upper_test(i,1) = mean2' * phi_basis( x_total(i) )' + 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
end
pred = mean2' * phi_basis(x_total)';
cover_fill(x_total*max_x_input,lower_test*max_target,upper_test*max_target,'g');
plot(x_total*max_x_input,pred*max_target,'k');

%% MSE
total_sum_test = 0;
total_sum_train = 0;

for i=1:size(x_new_index,1)
    if x_new_index(i)<=size(x_train,1)
        total_sum_train = total_sum_train + ( y_train( x_new_index(i) ) - pred(i) )^2;
    else
        total_sum_test = total_sum_test + ( y_test( x_new_index(i) - ( size(x_train,1)  ) ) - pred(i) )^2;
    end
end

MSEtrain(cnt_time) = total_sum_train / size(y_train,1);
% fprintf('Mean square error training set is: %f\n\n',MSEtrain(cnt_time));

MSEtest(cnt_time) = total_sum_test / size(y_test,1);
fprintf('Mean square error test set is: %f\n\n',MSEtest(cnt_time));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 4th degree
phi_basis = @(x) [x.^0 x.^1 x.^2 x.^3 x.^4];
num_of_basis = 5;

%% DESIGN MATRIX (POLYNOMIAL BASIS)
phi = phi_basis(x_train);

%% Alpha, Beta Parameters for Polynomial
[alpha,beta] = learning_hyper_parameters(x_train,y_train,phi_basis);

%% COVARIANCE -- MEAN polynomial
covariance = inv((alpha * eye(size(phi,2)) + beta * phi' * phi ));
mean2 = beta * covariance * phi' * y_train;

%% plots showing the true function, the data points, and the mean and variance of predictive distribution POLYNOMIAL
% cover_fill = @(x,lower,upper,color) set(fill([x;x(end:-1:1,:)],[lower;upper(end:-1:1,:)],color),'EdgeColor',color,'facealpha',0.125);

[x_total,x_new_index] = sortrows([x_train;x_test]);

for i=1:size(x_total,1)
    lower_test(i,1) = mean2' * phi_basis( x_total(i) )' - 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
    upper_test(i,1) = mean2' * phi_basis( x_total(i) )' + 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
end
pred = mean2' * phi_basis(x_total)';
% cover_fill(x_total,lower_test,upper_test,'g');
% plot(x_total,pred,'k');

%% MSE
total_sum_test = 0;
total_sum_train = 0;

for i=1:size(x_new_index,1)
    if x_new_index(i)<=size(x_train,1)
        total_sum_train = total_sum_train + ( y_train( x_new_index(i) ) - pred(i) )^2;
    else
        total_sum_test = total_sum_test + ( y_test( x_new_index(i) - ( size(x_train,1)  ) ) - pred(i) )^2;
    end
end

MSEtrain(cnt_time) = total_sum_train / size(y_train,1);
% fprintf('Mean square error training set is: %f\n\n',MSEtrain(cnt_time));

MSEtest(cnt_time) = total_sum_test / size(y_test,1);
fprintf('Mean square error test set is: %f\n\n',MSEtest(cnt_time));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 5th degree
phi_basis = @(x) [x.^0 x.^1 x.^2 x.^3 x.^4 x.^5];
num_of_basis = 6;

%% DESIGN MATRIX (POLYNOMIAL BASIS)
phi = phi_basis(x_train);

%% Alpha, Beta Parameters for Polynomial
[alpha,beta] = learning_hyper_parameters(x_train,y_train,phi_basis);

%% COVARIANCE -- MEAN polynomial
covariance = inv((alpha * eye(size(phi,2)) + beta * phi' * phi ));
mean2 = beta * covariance * phi' * y_train;

%% plots showing the true function, the data points, and the mean and variance of predictive distribution POLYNOMIAL
% cover_fill = @(x,lower,upper,color) set(fill([x;x(end:-1:1,:)],[lower;upper(end:-1:1,:)],color),'EdgeColor',color,'facealpha',0.125);

[x_total,x_new_index] = sortrows([x_train;x_test]);

for i=1:size(x_total,1)
    lower_test(i,1) = mean2' * phi_basis( x_total(i) )' - 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
    upper_test(i,1) = mean2' * phi_basis( x_total(i) )' + 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
end
pred = mean2' * phi_basis(x_total)';
% cover_fill(x_total,lower_test,upper_test,'g');
% plot(x_total,pred,'k');

%% MSE
total_sum_test = 0;
total_sum_train = 0;

for i=1:size(x_new_index,1)
    if x_new_index(i)<=size(x_train,1)
        total_sum_train = total_sum_train + ( y_train( x_new_index(i) ) - pred(i) )^2;
    else
        total_sum_test = total_sum_test + ( y_test( x_new_index(i) - ( size(x_train,1)  ) ) - pred(i) )^2;
    end
end

MSEtrain(cnt_time) = total_sum_train / size(y_train,1);
% fprintf('Mean square error training set is: %f\n\n',MSEtrain(cnt_time));

MSEtest(cnt_time) = total_sum_test / size(y_test,1);
fprintf('Mean square error test set is: %f\n\n',MSEtest(cnt_time));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 6th degree
phi_basis = @(x) [x.^0 x.^1 x.^2 x.^3 x.^4 x.^5 x.^6];
num_of_basis = 7;

%% DESIGN MATRIX (POLYNOMIAL BASIS)
phi = phi_basis(x_train);

%% Alpha, Beta Parameters for Polynomial
[alpha,beta] = learning_hyper_parameters(x_train,y_train,phi_basis);

%% COVARIANCE -- MEAN polynomial
covariance = inv((alpha * eye(size(phi,2)) + beta * phi' * phi ));
mean2 = beta * covariance * phi' * y_train;

%% plots showing the true function, the data points, and the mean and variance of predictive distribution POLYNOMIAL
% cover_fill = @(x,lower,upper,color) set(fill([x;x(end:-1:1,:)],[lower;upper(end:-1:1,:)],color),'EdgeColor',color,'facealpha',0.125);

[x_total,x_new_index] = sortrows([x_train;x_test]);

for i=1:size(x_total,1)
    lower_test(i,1) = mean2' * phi_basis( x_total(i) )' - 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
    upper_test(i,1) = mean2' * phi_basis( x_total(i) )' + 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
end
pred = mean2' * phi_basis(x_total)';
% cover_fill(x_total,lower_test,upper_test,'g');
% plot(x_total,pred,'k');

%% MSE
total_sum_test = 0;
total_sum_train = 0;

for i=1:size(x_new_index,1)
    if x_new_index(i)<=size(x_train,1)
        total_sum_train = total_sum_train + ( y_train( x_new_index(i) ) - pred(i) )^2;
    else
        total_sum_test = total_sum_test + ( y_test( x_new_index(i) - ( size(x_train,1)  ) ) - pred(i) )^2;
    end
end

MSEtrain(cnt_time) = total_sum_train / size(y_train,1);
% fprintf('Mean square error training set is: %f\n\n',MSEtrain(cnt_time));

MSEtest(cnt_time) = total_sum_test / size(y_test,1);
fprintf('Mean square error test set is: %f\n\n',MSEtest(cnt_time));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% BASIS POLYNOMIAL 7th degree
phi_basis = @(x) [x.^0 x.^1 x.^2 x.^3 x.^4 x.^5 x.^6 x.^7];
num_of_basis = 8;

%% DESIGN MATRIX (POLYNOMIAL BASIS)
phi = phi_basis(x_train);

%% Alpha, Beta Parameters for Polynomial
[alpha,beta] = learning_hyper_parameters(x_train,y_train,phi_basis);
% fprintf('Optimized parameters are: [%f %f]\n\n',alpha, beta)

%% COVARIANCE -- MEAN polynomial
covariance = inv((alpha * eye(size(phi,2)) + beta * phi' * phi ));
mean = beta * covariance * phi' * y_train;

%% plots showing the true function, the data points, and the mean and variance of predictive distribution POLYNOMIAL
% cover_fill = @(x,lower,upper,color) set(fill([x;x(end:-1:1,:)],[lower;upper(end:-1:1,:)],color),'EdgeColor',color,'facealpha',0.125);

[x_total,x_new_index] = sortrows([x_train;x_test]);

for i=1:size(x_total,1)
    lower_test(i,1) = mean' * phi_basis( x_total(i) )' - 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
    upper_test(i,1) = mean' * phi_basis( x_total(i) )' + 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
end
pred = mean' * phi_basis(x_total)';

%% SAMPLE BAYESIAN REGRESSION FUNCTION
w=zeros(5,num_of_basis);
y_out=zeros(5,N);
x_input_srt=sort(x_input);

for i=1:num_of_basis
   for k=1:5
      w(k,i) = normrnd(mean(i),covariance(i,i));
   end
end

for i=1:N
    for t=1:5
        y_out(t,i) = w(t,:) * phi_basis(x_input_srt(i))';
    end
end

% plot(x_input,target,'ro');
% plot(x_input_srt,y_out);
%% MSE
total_sum_test = 0;
total_sum_train = 0;

for i=1:size(x_new_index,1)
    if x_new_index(i)<=size(x_train,1)
        total_sum_train = total_sum_train + ( y_train( x_new_index(i) ) - pred(i) )^2;
    else
        total_sum_test = total_sum_test + ( y_test( x_new_index(i) - ( size(x_train,1)  ) ) - pred(i) )^2;
    end
end

MSEtrain(cnt_time) = total_sum_train / size(y_train,1);
% fprintf('Mean square error training set is: %f\n\n',MSEtrain(cnt_time));

MSEtest(cnt_time) = total_sum_test / size(y_test,1);
fprintf('Mean square error test set is: %f\n\n',MSEtest(cnt_time));


cover_fill(x_total*max_x_input,lower_test*max_target,upper_test*max_target,'b');
plot(x_total*max_x_input,pred*max_target,'r');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 8th degree

phi_basis = @(x) [x.^0 x.^1 x.^2 x.^3 x.^4 x.^5 x.^6 x.^7 x.^8];
num_of_basis = 9;

%% DESIGN MATRIX (POLYNOMIAL BASIS)
phi = phi_basis(x_train);

%% Alpha, Beta Parameters for Polynomial
[alpha,beta] = learning_hyper_parameters(x_train,y_train,phi_basis);

%% COVARIANCE -- MEAN polynomial
covariance = inv((alpha * eye(size(phi,2)) + beta * phi' * phi ));
mean3 = beta * covariance * phi' * y_train;

%% plots showing the true function, the data points, and the mean and variance of predictive distribution POLYNOMIAL
% cover_fill = @(x,lower,upper,color) set(fill([x;x(end:-1:1,:)],[lower;upper(end:-1:1,:)],color),'EdgeColor',color,'facealpha',0.125);

[x_total,x_new_index] = sortrows([x_train;x_test]);

for i=1:size(x_total,1)
    lower_test(i,1) = mean3' * phi_basis( x_total(i) )' - 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
    upper_test(i,1) = mean3' * phi_basis( x_total(i) )' + 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
end
% cover_fill(x_total,lower_test,upper_test,'m');

pred = mean3' * phi_basis(x_total)';
% plot(x_total,pred,'c');
% title('Bayesian Linear Regression Polynomial Basis Function'); 
% legend('training points','test points','variance of polyn. 7','mean of polyn. 7','variance of polyn. 2','mean of polyn. 2','variance of polyn. 12','mean of polyn. 12'); hold off;
% xlabel('price (10^{-3} / kWh)');
% ylabel('load consumption (kWh)');
% hold off

%% MSE
total_sum_test = 0;
total_sum_train = 0;

for i=1:size(x_new_index,1)
    if x_new_index(i)<=size(x_train,1)
        total_sum_train = total_sum_train + ( y_train( x_new_index(i) ) - pred(i) )^2;
    else
        total_sum_test = total_sum_test + ( y_test( x_new_index(i) - ( size(x_train,1)  ) ) - pred(i) )^2;
    end
end

MSEtrain(cnt_time) = total_sum_train / size(y_train,1);
% fprintf('Mean square error training set is: %f\n\n',MSEtrain(cnt_time));

MSEtest(cnt_time) = total_sum_test / size(y_test,1);
fprintf('Mean square error test set is: %f\n\n',MSEtest(cnt_time));

avg_mean_square_error_train = sum(MSEtrain)/length(MSEtrain);
avg_mean_square_error_test = sum(MSEtest)/length(MSEtest);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 9th degree

phi_basis = @(x) [x.^0 x.^1 x.^2 x.^3 x.^4 x.^5 x.^6 x.^7 x.^8 x.^9];
num_of_basis = 10;

%% DESIGN MATRIX (POLYNOMIAL BASIS)
phi = phi_basis(x_train);

%% Alpha, Beta Parameters for Polynomial
[alpha,beta] = learning_hyper_parameters(x_train,y_train,phi_basis);

%% COVARIANCE -- MEAN polynomial
covariance = inv((alpha * eye(size(phi,2)) + beta * phi' * phi ));
mean3 = beta * covariance * phi' * y_train;

%% plots showing the true function, the data points, and the mean and variance of predictive distribution POLYNOMIAL
% cover_fill = @(x,lower,upper,color) set(fill([x;x(end:-1:1,:)],[lower;upper(end:-1:1,:)],color),'EdgeColor',color,'facealpha',0.125);

[x_total,x_new_index] = sortrows([x_train;x_test]);

for i=1:size(x_total,1)
    lower_test(i,1) = mean3' * phi_basis( x_total(i) )' - 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
    upper_test(i,1) = mean3' * phi_basis( x_total(i) )' + 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
end
% cover_fill(x_total,lower_test,upper_test,'m');

pred = mean3' * phi_basis(x_total)';
% plot(x_total,pred,'c');
% title('Bayesian Linear Regression Polynomial Basis Function'); 
% legend('training points','test points','variance of polyn. 7','mean of polyn. 7','variance of polyn. 2','mean of polyn. 2','variance of polyn. 12','mean of polyn. 12'); hold off;
% xlabel('price (10^{-3} / kWh)');
% ylabel('load consumption (kWh)');
% hold off

%% MSE
total_sum_test = 0;
total_sum_train = 0;

for i=1:size(x_new_index,1)
    if x_new_index(i)<=size(x_train,1)
        total_sum_train = total_sum_train + ( y_train( x_new_index(i) ) - pred(i) )^2;
    else
        total_sum_test = total_sum_test + ( y_test( x_new_index(i) - ( size(x_train,1)  ) ) - pred(i) )^2;
    end
end

MSEtrain(cnt_time) = total_sum_train / size(y_train,1);
% fprintf('Mean square error training set is: %f\n\n',MSEtrain(cnt_time));

MSEtest(cnt_time) = total_sum_test / size(y_test,1);
fprintf('Mean square error test set is: %f\n\n',MSEtest(cnt_time));

avg_mean_square_error_train = sum(MSEtrain)/length(MSEtrain);
avg_mean_square_error_test = sum(MSEtest)/length(MSEtest);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 10th degree

phi_basis = @(x) [x.^0 x.^1 x.^2 x.^3 x.^4 x.^5 x.^6 x.^7 x.^8 x.^9 x.^10];
num_of_basis = 11;

%% DESIGN MATRIX (POLYNOMIAL BASIS)
phi = phi_basis(x_train);

%% Alpha, Beta Parameters for Polynomial
[alpha,beta] = learning_hyper_parameters(x_train,y_train,phi_basis);

%% COVARIANCE -- MEAN polynomial
covariance = inv((alpha * eye(size(phi,2)) + beta * phi' * phi ));
mean3 = beta * covariance * phi' * y_train;

%% plots showing the true function, the data points, and the mean and variance of predictive distribution POLYNOMIAL
% cover_fill = @(x,lower,upper,color) set(fill([x;x(end:-1:1,:)],[lower;upper(end:-1:1,:)],color),'EdgeColor',color,'facealpha',0.125);

[x_total,x_new_index] = sortrows([x_train;x_test]);

for i=1:size(x_total,1)
    lower_test(i,1) = mean3' * phi_basis( x_total(i) )' - 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
    upper_test(i,1) = mean3' * phi_basis( x_total(i) )' + 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
end
cover_fill(x_total*max_x_input,lower_test*max_target,upper_test*max_target,'m');

pred = mean3' * phi_basis(x_total)';
plot(x_total*max_x_input,pred*max_target,'c');
title('Bayesian Linear Regression Polynomial Basis Function'); 
legend('(input,target)','training points','test points','variance of polyn. 3','mean of polyn. 3','variance of polyn. 7','mean of polyn. 7','variance of polyn. 10','mean of polyn. 10'); 
hold off;
xlabel('price per kWh');
ylabel('kWh');
hold off

%% MSE
total_sum_test = 0;
total_sum_train = 0;

for i=1:size(x_new_index,1)
    if x_new_index(i)<=size(x_train,1)
        total_sum_train = total_sum_train + ( y_train( x_new_index(i) ) - pred(i) )^2;
    else
        total_sum_test = total_sum_test + ( y_test( x_new_index(i) - ( size(x_train,1)  ) ) - pred(i) )^2;
    end
end

MSEtrain(cnt_time) = total_sum_train / size(y_train,1);
% fprintf('Mean square error training set is: %f\n\n',MSEtrain(cnt_time));

MSEtest(cnt_time) = total_sum_test / size(y_test,1);
fprintf('Mean square error test set is: %f\n\n',MSEtest(cnt_time));

avg_mean_square_error_train = sum(MSEtrain)/length(MSEtrain);
avg_mean_square_error_test = sum(MSEtest)/length(MSEtest);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 11th degree

phi_basis = @(x) [x.^0 x.^1 x.^2 x.^3 x.^4 x.^5 x.^6 x.^7 x.^8 x.^9 x.^10 x.^11];
num_of_basis = 12;

%% DESIGN MATRIX (POLYNOMIAL BASIS)
phi = phi_basis(x_train);

%% Alpha, Beta Parameters for Polynomial
[alpha,beta] = learning_hyper_parameters(x_train,y_train,phi_basis);

%% COVARIANCE -- MEAN polynomial
covariance = inv((alpha * eye(size(phi,2)) + beta * phi' * phi ));
mean3 = beta * covariance * phi' * y_train;

%% plots showing the true function, the data points, and the mean and variance of predictive distribution POLYNOMIAL
% cover_fill = @(x,lower,upper,color) set(fill([x;x(end:-1:1,:)],[lower;upper(end:-1:1,:)],color),'EdgeColor',color,'facealpha',0.125);

[x_total,x_new_index] = sortrows([x_train;x_test]);

for i=1:size(x_total,1)
    lower_test(i,1) = mean3' * phi_basis( x_total(i) )' - 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
    upper_test(i,1) = mean3' * phi_basis( x_total(i) )' + 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
end
% cover_fill(x_total,lower_test,upper_test,'m');

pred = mean3' * phi_basis(x_total)';
% plot(x_total,pred,'c');
% title('Bayesian Linear Regression Polynomial Basis Function'); 
% legend('training points','test points','variance of polyn. 7','mean of polyn. 7','variance of polyn. 2','mean of polyn. 2','variance of polyn. 12','mean of polyn. 12'); hold off;
% xlabel('price (10^{-3} / kWh)');
% ylabel('load consumption (kWh)');
% hold off

%% MSE
total_sum_test = 0;
total_sum_train = 0;

for i=1:size(x_new_index,1)
    if x_new_index(i)<=size(x_train,1)
        total_sum_train = total_sum_train + ( y_train( x_new_index(i) ) - pred(i) )^2;
    else
        total_sum_test = total_sum_test + ( y_test( x_new_index(i) - ( size(x_train,1)  ) ) - pred(i) )^2;
    end
end

MSEtrain(cnt_time) = total_sum_train / size(y_train,1);
% fprintf('Mean square error training set is: %f\n\n',MSEtrain(cnt_time));

MSEtest(cnt_time) = total_sum_test / size(y_test,1);
fprintf('Mean square error test set is: %f\n\n',MSEtest(cnt_time));

avg_mean_square_error_train = sum(MSEtrain)/length(MSEtrain);
avg_mean_square_error_test = sum(MSEtest)/length(MSEtest);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% 12th degree

phi_basis = @(x) [x.^0 x.^1 x.^2 x.^3 x.^4 x.^5 x.^6 x.^7 x.^8 x.^9 x.^10 x.^11 x.^12];
num_of_basis = 10;

%% DESIGN MATRIX (POLYNOMIAL BASIS)
phi = phi_basis(x_train);

%% Alpha, Beta Parameters for Polynomial
[alpha,beta] = learning_hyper_parameters(x_train,y_train,phi_basis);

%% COVARIANCE -- MEAN polynomial
covariance = inv((alpha * eye(size(phi,2)) + beta * phi' * phi ));
mean3 = beta * covariance * phi' * y_train;

%% plots showing the true function, the data points, and the mean and variance of predictive distribution POLYNOMIAL
% cover_fill = @(x,lower,upper,color) set(fill([x;x(end:-1:1,:)],[lower;upper(end:-1:1,:)],color),'EdgeColor',color,'facealpha',0.125);

[x_total,x_new_index] = sortrows([x_train;x_test]);

for i=1:size(x_total,1)
    lower_test(i,1) = mean3' * phi_basis( x_total(i) )' - 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
    upper_test(i,1) = mean3' * phi_basis( x_total(i) )' + 2*sqrt( (1/beta) + phi_basis(x_total(i)) * covariance * phi_basis( x_total(i) )' );
end
% cover_fill(x_total,lower_test,upper_test,'m');

pred = mean3' * phi_basis(x_total)';
% plot(x_total,pred,'c');
% title('Bayesian Linear Regression Polynomial Basis Function'); 
% legend('training points','test points','variance of polyn. 7','mean of polyn. 7','variance of polyn. 2','mean of polyn. 2','variance of polyn. 12','mean of polyn. 12'); hold off;
% xlabel('price (10^{-3} / kWh)');
% ylabel('load consumption (kWh)');
% hold off

%% MSE
total_sum_test = 0;
total_sum_train = 0;

for i=1:size(x_new_index,1)
    if x_new_index(i)<=size(x_train,1)
        total_sum_train = total_sum_train + ( y_train( x_new_index(i) ) - pred(i) )^2;
    else
        total_sum_test = total_sum_test + ( y_test( x_new_index(i) - ( size(x_train,1)  ) ) - pred(i) )^2;
    end
end

MSEtrain(cnt_time) = total_sum_train / size(y_train,1);
% fprintf('Mean square error training set is: %f\n\n',MSEtrain(cnt_time));

MSEtest(cnt_time) = total_sum_test / size(y_test,1);
fprintf('Mean square error test set is: %f\n\n',MSEtest(cnt_time));

avg_mean_square_error_train = sum(MSEtrain)/length(MSEtrain);
avg_mean_square_error_test = sum(MSEtest)/length(MSEtest);