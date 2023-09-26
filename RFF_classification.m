%% Test one_hot_matrix

y = linspace(0,9,10);
A = one_hot_matrix(y);

%% Solve the lsq-problems

% Parameters
J = 10000;
M = 1;
num_mistaken = zeros(M,1);
percentage_mistaken = zeros(M,1);
var = 0.1;
lambda = 0.02;
lambda_vec = zeros(M,1);

for k = 1:M
    
    %K = 2^k
    K = 800;
    
    % Import data from MNIST dataset
    [train_images, train_labels] = ...
        readMNIST("data/train-images-idx3-ubyte", ...
        "data/train-labels-idx1-ubyte", J, 0);
    [test_images, test_labels] = ...
        readMNIST("data/t10k-images-idx3-ubyte", ...
        "data/t10k-labels-idx1-ubyte", J, 0);
    
    % Create random Fourier features.
    omega = sqrt(var)*randn(784, K);
    x_train = reshape(train_images, 784, []);
    S = exp(1i * x_train' * omega);
    y_train = one_hot_matrix(train_labels);
    
    %lambda = 0.01*2^k;
    %lambda_vec(k) = lambda
    
    % Solve the 10 lsq problems (6)
    
    beta = zeros(K, 10);
    for i = 1:10
        y_i = y_train(:, i); % y_i is the ith column in y_train
        
        A = (S'*S + lambda*J*eye(K));
        b = (S'*y_i);
     
        beta(:, i) = A \ b; % Solve normal eqn: Ax = b.
    end
    
    % Compute the percentage of mistaken digits.
    
    x_test = reshape(test_images, 784, []);
    S_test = exp(1i*x_test'*omega);
    
    mag_beta = abs(beta'*S_test');
    
    [~, idx] = max(mag_beta, [], 1);
    idx = idx - 1;
    
    num_mistaken(k) = sum(test_labels' ~= idx);
    
    percentage_mistaken(k) = (num_mistaken(k) / length(test_labels))*100;
end

%% Recreate the plot of the generalization error as a function of K

k_vec = zeros(M,1);
for j = 1:M
    k_vec(j) = 2^j;
end

% Plot generalization error as function of K

%figure();
%loglog(k_vec, 100*k_vec.^(-0.5), Color='r');
%hold on
%scatter(k_vec, percentage_mistaken,  '*', 'b')
%hold on
%title('Generalization error of RFF on MNIST')
%xlabel('K')
%ylabel('Percent mistaken')

% Plot generalization error as function of lambda
%figure();
%scatter(lambda_vec, percentage_mistaken,  '*', 'k')
%title('Generalization error as function of \lambda')
%xlabel('\lambda')
%ylabel('Generalization error')

%% Check the frequency of digits in test data

M1 = mode(test_labels)
[n, bin] = hist(test_labels, unique(test_labels))
[~, id] = sort(-n)
disp(n(id))
bin(id) % most common numbers
sorted_digits = bin(id);

% Weighted frequencies
new_covariance = zeros(784);
for digit = 1:10
    new_covariance = new_covariance + ...
        + omega(:, digit) * omega(:, digit)' * n(digit) / sum(n);
end

% Generate new omega matrix
new_covariance = new_covariance / norm(new_covariance, 'fro');
new_stddevs = sqrt(diag(new_covariance));
normal_sample = randn(784, K);
new_omega = diag(new_stddevs)*normal_sample;


%% Solve the lsq problems with new frequencies

S = exp(1i * x_train' * new_omega);

beta = zeros(K, 10);
for i = 1:10
    y_i = y_train(:, i); % y_i is the ith column in y_train
        
    A = (S'*S + lambda*J*eye(K));
    b = (S'*y_i);
     
    beta(:, i) = A \ b; % Solve normal eqn: Ax = b.
end
    
% Compute the percentage of mistaken digits.
S_test = exp(1i*x_test'*new_omega);
mag_beta = abs(beta'*S_test');
    
[~, idx] = max(mag_beta, [], 1);
idx = idx - 1;
    
num_mistaken_freq = sum(test_labels' ~= idx);
    
percentage_mistaken_freq = (num_mistaken_freq / length(test_labels))*100;

