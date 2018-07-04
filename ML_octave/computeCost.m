function J = computeCost(X, y, theta)
% COMPUTECOST Compute cost for linear regression
%  J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%  parameter for linear regression to fit the data points in X and y

% Initialization
m = length(y); % number of training examples
J = 0;         % return cost

%  Compute the cost

h_theta = X*theta;  % hypothesis, dimensions m*1 = m*n x n*1
J = 1/(2*m)*sum((h_theta-y).^2);

end