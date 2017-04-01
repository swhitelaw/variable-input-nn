function [error_train, error_val] = ...
    learningCurve(X_1, y_1, X_2, y_2, Xval_1, yval_1, Xval_2, yval_2, lambda,input_layer_size, hidden_layer_size, output_layer_size)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m_1 = size(X_1, 1);
m_2 = size(X_2, 1);
m = min(m_2,m_1);
% You need to return these values correctly
error_train = zeros(floor(m/10), 1);
error_val   = zeros(floor(m/10), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------

for i = 1:floor(m/10),
	X_train_1 = X_1(1:i*10, :);
	y_train_1 = y_1(1:i*10);
	X_train_2 = X_2(1:i*10, :);
	y_train_2 = y_2(1:i*10);

	[nn_params, cost] = trainLinearReg(X_train_1,y_train_1,X_train_2,y_train_2,lambda,input_layer_size, hidden_layer_size, output_layer_size);
	error_train(i) = nnCostFunction(nn_params,input_layer_size, hidden_layer_size, output_layer_size, X_train_1,y_train_1,X_train_2,y_train_2, 0);

	error_val(i) = nnCostFunction(nn_params,input_layer_size, hidden_layer_size, output_layer_size,Xval_1, yval_1, Xval_2, yval_2, 0);
end





% -------------------------------------------------------------

% =========================================================================

end
