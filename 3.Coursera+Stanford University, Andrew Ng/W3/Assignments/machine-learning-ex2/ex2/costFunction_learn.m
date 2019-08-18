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
%Compute J using vectorization

%z = X*theta; %size(X) = m 3; size(theta) = 3 1; size(z) = m 1
%hypothesis = 1 ./(1+ exp(-z)); %size(hipothesis) = m 1
%A1 = log(hypothesis); %size(A1) = m 1
%A2 = log(1-hypothesis);%size(A2) = m 1
%T1 = (-1/m)*y'*A1;  %first component of the cost function vectorized; size(y') = 1 m, size(A1) = m 1; size(y'*A1) = 1 (scalar)
%T2 = (-1/m)*(ones(m,1)-y')*A2; %second component of the cost function vectorized; size(y') = 1 m, size(A2) = m 1; size(y'*A2) = 1 (scalar)

%J = T1 + T2; %size(J) = 1 (scalar)

J = (-1/m)*[y'*log(1 ./(1 + exp(-1*X*theta))) + (1-y')*log(1- 1 ./(1 + exp(-1*X*theta)))];

%Compute grad:

error = X*theta -y; %size(X) = m n+1, size(theta) = n+1 1, size(X*theta) = m 1, size(y) = m 1, size(error) = m 1
grad = (1/m)*(X'*error); % size(X') = n+1 m size(error) = m 1 => size(X'*error) = n+1 1; size(theta) = n+1 1 => size(grad) = size(theta) = n+1 1 => OK










% =============================================================

end
