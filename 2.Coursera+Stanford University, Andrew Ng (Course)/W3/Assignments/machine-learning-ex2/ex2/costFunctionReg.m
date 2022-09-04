function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta),1);
n = length(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%AP: Compute J:
hypothesis = sigmoid(X*theta); %size(hypothesis) = m 1

regterm = theta([2:size(theta)]); %we'll apply regularization starting from theta2; indices are translated with 1 in the application vs formulas!!!
regterm = regterm .^2;
J = (-1/m)*sum((y'*log(hypothesis) + (1-y')*log(1- hypothesis))) + (lambda/(2*m))*sum(regterm); %size J = 1 1( scalar)

%compute gradient using vectorization
error = hypothesis - y; %size(error) = m 1
grad1 = (1/m)*(X'(1,:)*error); %size(X'(1,:)) = 1 m; size(error) = m 1 => size(grad1) = 1 1(scalar)
%fprintf("Size hypothesis = %f\t, Size X = %f\t,  Size X' = %f\t, Size y = %f\t,Size(error) = %f\t, size(theta) = %f\t", size(hypothesis), size(X), size(X'),size(y), size(error), size(theta));
gradrest = (1/m)*((X'(2:n,:)*error)) + (lambda/m)*theta(2:n,:);
grad = [grad1;gradrest];


% =============================================================

end
