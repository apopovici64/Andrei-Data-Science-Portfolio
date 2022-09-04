function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
%AP: Compute J:
hypothesis = sigmoid(X*theta); %size(hypothesis) = m 1

regterm = theta([2:size(theta)]); %we'll apply regularization starting from theta2; indices are translated with 1 in the application vs formulas!!!
regterm = regterm .^2;
J = (-1/m)*sum((y'*log(hypothesis) + (1-y')*log(1- hypothesis))) + (lambda/(2*m))*sum(regterm); %size J = 1 1( scalar)

%compute gradient using vectorization
error = hypothesis - y; %size(error) = m 1
grad1 = (1/m)*(X'(1,:)*error); %size(X'(1,:)) = 1 m; size(error) = m 1 => size(grad1) = 1 1(scalar)
gradrest = (1/m)*((X'(2:n,:)*error)) + (lambda/m)*theta(2:n,:);
grad = [grad1;gradrest];


% =============================================================

end

