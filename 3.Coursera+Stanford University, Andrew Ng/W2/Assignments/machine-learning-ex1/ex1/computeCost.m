function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

predictions = X*theta; %predictions of hypothesis of all m examples; size(X) = m 2, size(theta) = 2 1, size(predictions) = m 1
errors = predictions - y; %size(predictions) = m 1, size(y) = m 1, size(errors) = m 1
sqrErrors = errors.^2; %squared errors; size(sqrErrors) = m 1

J = (1/(2*m)) * sum(sqrErrors,1); %compute J as the cost function by summing its elements  per column



% =========================================================================

endfunction
