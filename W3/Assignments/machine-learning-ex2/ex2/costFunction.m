function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%AP: Compute J using vectorization:

J = (-1/m)*(y'*log(1 ./(1 + exp(-1*X*theta))) + (1-y')*log(1- 1 ./(1 + exp(-1*X*theta)))); %size(J) = 1 1 (scalar)

%AP:Compute grad using vectorization:
hypothesis = 1 ./(1 + exp(-1*X*theta)); %size(hypothesis) = m 1
error = hypothesis - y; %error is always different depending on the function hypothesis, here we have hypothesis = sigmoid function, not linear; size(error) = m 1
grad = (1/m)*(X'*error); %grad in this case is identical as formula with the one from linear regresion; size(X') = n+1 m size(error) = m 1 => size(X'*error) = n+1 1; size(theta) = n+1 1 => size(grad) = size(theta) = n+1 1 => OK










% =============================================================

end
