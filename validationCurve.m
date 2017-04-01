function [lambda_vec, error_train, error_val] = ...
    validationCurve(X_1, y_1, X_2, y_2, Xval_1, yval_1, Xval_2, yval_2, input_layer_size, hidden_layer_size, output_layer_size)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.01 0.03 0.1 0.3 1 3]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%
for i=1:length(lambda_vec),
	[nn_params, cost] = trainLinearReg(X_1, y_1, X_2, y_2, lambda_vec(i),input_layer_size, hidden_layer_size, output_layer_size);
	error_train(i) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,output_layer_size, X_1, y_1, X_2, y_2, 0);
	error_val(i) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,output_layer_size,Xval_1, yval_1, Xval_2, yval_2, 0);
end









% =========================================================================

end
