%% Linear Regression example in Matlab
clear ; close all; clc
% m ... number of training examples
% n ... number of features
%% Load and plot data 
fprintf('Plotting Data ...\n')
data = load('ex1data2.txt');
% x1 refers to size of house in square feet
% x2 refers to # bedrooms
X = data(:, 1:2);
% y refers to house price
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

figure(1);
plot(X(:,1),(y./1e4),'rx'); hold on;
title('Housing prices vs. size in Portland, Oregon')
xlabel('Size of the house (in square feet)'); % Set the x-axis label
ylabel('House price in $10,000s'); % Set the y-axis label
figure(2);
plot(X(:,2),y./1e4,'bx');
title('Housing prices vs. number of bedrooms in Portland, Oregon')
xlabel('Number of bedrooms'); % Set the x-axis label
ylabel('House price in $10,000s'); % Set the y-axis label

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featNorm(X);

% Add intercept term to X
X = [ones(m, 1) X];

%% Gradient descent
fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.05;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% the first column of X is all-ones. Thus, it does not need to be normalized.
X_p = [1650, 3];
X_p_norm = (X_p-mu)./sigma;
X_op_norm = [1, X_p_norm];
price = X_op_norm*theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);
%% Normal Equations

fprintf('Solving with normal equations...\n');

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
price = [1, 1650, 3]*theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

