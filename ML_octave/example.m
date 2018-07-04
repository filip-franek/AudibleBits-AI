%% Linear Regression example in Matlab
clear ; close all; clc
% m ... number of training examples
% n ... number of features
%% Load and plot data 
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');

% x refers to the population size in 10,000s
X_mx1 = data(:, 1);
% y refers to the profit in $10,000s
y_mx1 = data(:, 2);

% X_mx1 = [34,108,64,88,99,51]';
% y_mx1 = [5,17,11,8,14,5]';

m = length(y_mx1); % number of training examples

% Plot Data
figure(1);
plot(X_mx1,y_mx1,'rx', 'MarkerSize', 10)
title('Data')
xlabel('x_1'); % Set the x-axis label
ylabel('y'); % Set the y-axis label

X_mxn = [ones(m, 1), X_mx1]; % Add a column of ones to x
theta_nx1 = zeros(2, 1); % initialize fitting parameters

% Gradient descent
% - in batch gradient descent each iteration performs an update for all theta_j 
% settings
iterations = 1000;
alpha = 0.01;

% compute and display initial cost
fprintf('\nComputig the cost ...\n')
h_theta_mx1 = X_mxn*theta_nx1;  % hypothesis
J = 1/(2*m)*sum((h_theta_mx1-y_mx1).^2);
% J = computeCost(X, y, theta); % or this function

% run gradient descent
fprintf('\nRunning Gradient Descent ...\n')
J_hist = [J;zeros(iterations, 1)];
theta_hist = zeros(iterations, 2);

for iter = 1:iterations
    h_mx1 = X_mxn*theta_nx1;
    error_mx1 = h_mx1-y_mx1;
    theta_change_nx1 = alpha/m*(X_mxn'*error_mx1);
    theta_nx1 = theta_nx1 - theta_change_nx1;
    % Save the cost J in every iteration
    J_hist(iter+1) = computeCost(X_mxn, y_mx1, theta_nx1);
    theta_hist(iter+1,:) = theta_nx1';
end
%theta = gradientDescent(X, y, theta, alpha, iterations);

% Plot the linear fit
hold on
plot(X_mxn(:,2), X_mxn*theta_nx1, '-')
% legend('Training data', 'Linear regression')

% Predict values for a and b
x1 = mean(X_mx1);
x2 = min(X_mx1)+(max(X_mx1)-min(X_mx1))/100*90;
x3 = min(X_mx1)+(max(X_mx1)-min(X_mx1))/100*10;
predict_y1 = [1, x1] *theta_nx1;
fprintf('For x_1 = %f, we predict y = %f\n', x1, predict_y1);
predict_y2 = [1, x2] * theta_nx1;
fprintf('For x_2 = %f, we predict y = %f\n', x2, predict_y2);
predict_y3 = [1, x3] * theta_nx1;
fprintf('For x_3 = %f, we predict y = %f\n', x3, predict_y3);

plot([x1,x2,x3], [predict_y1,predict_y2,predict_y3], 'sk')
legend('Training data', 'Linear regression', 'Predicted values')

figure(2);
plot(J_hist)

%% Visualizing J(theta_0, theta_1)
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
% theta0_vals = linspace(theta_nx1(1)-(max(theta_hist(:,1))-min(theta_hist(:,1)))*1.5,...
%     theta_nx1(1)+(max(theta_hist(:,1))-min(theta_hist(:,1)))*1.5, 100);
% theta1_vals = linspace(theta_nx1(2)-(max(theta_hist(:,2))-min(theta_hist(:,2)))*1.5,...
%     theta_nx1(2)+(max(theta_hist(:,2))-min(theta_hist(:,2)))*1.5, 100);
% theta_nx1(1)-abs(theta_nx1(1))*100, theta_nx1(1)+abs(theta_nx1(1))*10
theta0_vals = linspace(-10,10,100);
theta1_vals = linspace(-4,4,100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X_mxn, y_mx1, t);
    end
end

% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure(3);
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals,logspace(-1, 2, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta_nx1(1), theta_nx1(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
plot(theta_hist(:,1),theta_hist(:,2),'-.')
hold off