function [y] = sample_gp_prior(k, X,theta)
  K = compute_kernel_matrix(k, X, X, theta);
  % sample from zero-mean Gaussian distribution with covariance matrix K
  [U,S,V] = svd(K);
  A = U*sqrt(S)*V';
  y = A * randn(size(X,1),1);
end 