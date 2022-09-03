function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%
%AP: prediction => decide if p(i, 1) in {0,1} based on the assesment of sigmoid (X*theta) >= 0.5 for all i = 1 : m
%AP: size(X) = m n+1, size(theta) = n+1 1, size(X*theta) = m 1 (vectorized approach for assesment)

prob = sigmoid(X*theta);%compute actual set of probabilities for the training set in X (having n+1 features)
for i = 1:m
  if prob(i, 1) >=0.5 
    %printf('Probability is: %f\n', prob(i,1)); TODO: save probabilities in a separate file.
    p(i,1) = 1;
  else 
    %printf('Probability is: %f\n', prob(i,1));
    p(i,1) = 0;
  endif
endfor
  






% =========================================================================


end
