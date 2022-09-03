function [theta J_hist] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples



for iter = 1:num_iters
    J_hist = zeros(num_iters, 1);
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================

    error = X*theta -y;
    theta = theta - (alpha/m)*(X'*error);
    
    %fprintf("Cost function is: %0.2f.\nThe gradient value is: %0.2f.\nThe iteration is: %d.\n", computeCost(X, theta),theta, iter); 
       
    % Save the cost J in every iteration    
    J_hist(iter,1) = computeCost(X, y, theta);
    save CostHistory.txt J_hist -ascii;
    
endfor

endfunction

