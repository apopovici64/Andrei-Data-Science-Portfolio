function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%Andrei Popovici comments:
%Notations: word/letter in all uppercase denotes a vector/matrice; word/letter in all lowercase denotes element in vector/matrice
%from ex1: THETA' = [theta0, theta1] (we have linear regression with a single variable, n = 1)

%THETA = theta; %theta0 = THETA(1,1)=THETA'(1,1), theta1 = THETA(2,1) = THETA'(1,2)
 
for iter = 1:num_iters
	
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	%Andrei Popovici comments:
	%Notations: word/letter in all uppercase denotes a vector/matrice; word/letter in all lowercase denotes element in vector/matrice
	%
	%from ex1: THETA' = theta' = [theta0, theta1] (change to uppercase notation; we have linear regression with a single variable, n = 1)
	%           X = [X0 X1] = [ones(m, 1), data(:,1)] (we'll take one line of this matrice corresponding to one training sample, where x0=1, x1=given from data; n =1)
	%           y = given from data, according to the convention above should be "Y" because is a vector with size(y)=(m,1); however, we'll keep the notation provided in the exercise
	%           
	%compute DELTA vector in the formula of GRADIENTDESCENT: THETA = THETA - alpha*DELTA; DELTA' = [delta0, delta1] 
	%delta0 = (1/m)*sum(g(1:m))*x0  where g(x0, x1) = (theta0*x0(i) + theta1*x1(i) - y(i)); x0(i)=1; i =(1:m)
	%delta1 = (1/m)*sum(g(1:m))*x1  where g(x0, x1) = (theta0*x0(i) + theta1*x1(i) - y(i)); x1(i)=1; i =(1:m) 
	printf("\n=======Iterating by num_iters: %d========\n", iter);
	THETA = theta;
		 for i = 1:m
			
			Gsample = zeros(2,1);
			Xsample = [X(i,1), X(i,2)];%X(i,1)=1 for i=1:m
			ysample = [y(i,1)];
			Gsample = Gsample +(THETA'*Xsample' - ysample)*Xsample'; 
			DELTAsample = (1/m)*Gsample;		
			THETA = THETA - alpha * DELTAsample; %apply one iteration update for THETA = [theta0;theta1]; could also use here one line formula: THETA = THETA - alpha * (1/m)*(THETA'*[1,X(i,2)] - [y(i,1)])	   
			
		 end	
    theta = THETA;    	
		
	fprintf('Cost function is: %0.2f.\nThe gradient value is: %0.2f.\nThe iteration is: %d.\n', computeCost(X, theta),theta, iter);  
	
	 

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	iter = iter + 1;

end

end
