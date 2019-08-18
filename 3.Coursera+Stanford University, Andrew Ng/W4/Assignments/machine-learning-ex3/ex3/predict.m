function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%AP compute the outputs for each layer of the NN:
a_1 = [ones(m, 1) X]; %add a0(1) vector to X matrix

z_2 = a_1 * Theta1'; %compute z(2) as product of a(1) and Theta1'  

a_2 = [ones(m,1) sigmoid(z_2)]; %compute a(2) output as fuction of sigmoid(z_2) + add a column of a0(2) = 1-bias

z_3 = a_2 * Theta2'; %compute z(3) as product of a(2) and Theta2' 

a_3 = sigmoid(z_3); %compute final output a(3) = h(x) as fuction of sigmoid(z_3)

%AP compute the  prediction as vector of probabilities
[highest_probability,p] = max(a_3, [] , 2);






% =========================================================================


end
