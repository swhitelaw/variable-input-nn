function p = predict(Theta1, Theta2, X, idx_to_elim)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
	m = size(X, 1);
	if idx_to_elim == 0,
		Theta1_1=Theta1;
	else,
		Theta1_1=Theta1(:,1:end-1);
	end

% You need to return the following variables correctly 
	p = zeros(size(X, 1), 1);

	h1 = sigmoid([ones(m, 1) X] * Theta1_1');
	h2 = sigmoid([ones(m, 1) h1] * Theta2');
	[dummy, p] = max(h2, [], 2);

	
% =========================================================================


end
