%% Machine Learning Online Class - Exercise 4 Neural Network Learning

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc
setenv('GNUTERM','qt')
%% Setup the parameters you will use for this exercise
input_layer_size  = 14;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
output_layer_size = 40;         % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
feats = 14;		%number of features
trials = 100;	%number of trials
%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

data_raw = csvread('leaf.csv');
m = size(data_raw, 1);
m_1 = floor(m/2);
lambda=0.03;

data  = data_raw(randperm(m),:);
[X mu sig] = featureNormalize(data(:, 3:end));
y=data(:,1);

X_train = X(1:.8*m, :);
X_cv = X(.8*m+1:.9*m, :);
X_test = X(.9*m+1:end, :);

m_train = size(X_train, 1);
m_1_train = floor(m_train/2);

m_cv = size(X_cv, 1);
m_1_cv = floor(m_cv/2);

y_train = y(1:.8*m, :);
y_cv = y(.8*m+1:.9*m, :);
y_test = y(.9*m+1:end, :);
%%
%%m_train = size(data_train, 1);
%%m_cv = size(data_cv, 1);
%%m_cv_1 = floor(m_cv/2);
%%
%%m_test = size(data_test, 1);
%%
%%%The full Data
%%X_train = data_train(:, 2:size(data, 2)-1);
%%y_train = data_train(:, size(data, 2):end)==4;
%%
%%% Training eliminating a feature
%%X_train_1 = X_train(1:m_1,2:end);
%%X_train_2 = [X_train(m_1+1:end,2:end) X_train(m_1+1:end,1)];
%%y_train_1 = y_train(1:m_1);
%%y_train_2 = y_train(m_1+1:end);
%%
%%X_cv = data_cv(:, 2:size(data, 2)-1);
%%y_cv = data_cv(:, size(data, 2):end)==4;
%%
%%X_cv_1 = X_cv(1:m_cv_1,2:end);
%%X_cv_2 = [X_cv(m_cv_1+1:end,2:end) X_cv(m_cv_1+1:end,1)];
%%y_cv_1 = y_cv(1:m_cv_1);
%%y_cv_2 = y_cv(m_cv_1+1:end);
%%
%%X_test = data_test(:, 2:size(data, 2)-1);
%%y_test = data_test(:, size(data, 2):end)==4;
%%
% find lambda Training with Full data
%[lambda_vec, error_train, error_val] = ...
%    validationCurve( [], [], X_train, y_train, [], [],X_cv, y_cv,input_layer_size, hidden_layer_size, output_layer_size);
%%%
%close all;
%plot(lambda_vec, error_train, lambda_vec, error_val);
%legend('Train', 'Cross Validation');
%xlabel('lambda');
%ylabel('Error');
%%
%%fprintf('lambda\t\tTrain Error\tValidation Error\n');
%%for i = 1:length(lambda_vec)
%%  fprintf(' %f\t%f\t%f\n', ...
%%            lambda_vec(i), error_train(i), error_val(i));
%%end
%%
%%
%%
%%pause;

% Compute Test Error


%% Full/ missing
%for j=1:9,
%accuracyTot = 0;
%for i=1:trials,
%%
%%	data  = data_raw(randperm(m),:);
%%	data_train = data(1:540, :);
%%	data_cv = data(541:600, :);
%%	data_test = data(601:699, :);
%%	
%%	m_train = size(data_train, 1);
%%	m_cv = size(data_cv, 1);
%%	m_test = size(data_test, 1);
%%	
%%	%The full Data
%%	X_train = data_train(:, 2:size(data, 2)-1);
%%	y_train = data_train(:, size(data, 2):end)==4;
%%	
%%	X_cv = data_cv(:, 2:size(data, 2)-1);
%%	y_cv = data_cv(:, size(data, 2):end)==4;
%%	
%%	X_test = data_test(:, 2:size(data, 2)-1);
%%	y_test = data_test(:, size(data, 2):end)==4;
%%
%%	%X_test = [X_test(:,1:j-1) X_test(:,j+1:end)];
%%	if j==1,
%%		X_test = X_test(:, 2:end);
%%	elseif j==9,
%%		X_test = X_test(:, 1:end-1);
%%	else
%%		X_test=[X_test(:, 1:j-1) X_test(:, j+1:end)];
%%	end
%%		
%
%data  = data_raw(randperm(m),:);
%[X mu sig] = featureNormalize(data(:, 3:end));
%y=data(:,1);
%
%X_train = X(1:.8*m, :);
%X_cv = X(.8*m+1:.9*m, :);
%X_test = X(.9*m:end, :);
%
%y_train = y(1:.8*m, :);
%y_cv = y(.8*m+1:.9*m, :);
%y_test = y(.9*m:end, :);
%
%	[nn_params, cost] = trainLinearReg([], [], X_train, y_train, lambda,input_layer_size, hidden_layer_size, output_layer_size);
%
%
%	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                	hidden_layer_size, (input_layer_size + 1));
%
%	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                	output_layer_size, (hidden_layer_size + 1));
%
%	pred = predict(Theta1, Theta2, X_test,0);
%
%
%	accuracy = mean(double(pred == y_test));
%	accuracyTot = accuracyTot+accuracy;
%%end
%%
%%j
%fprintf('\nTraining Set Accuracy: %f\n', accuracy * 100);
%end
%fprintf('\nAvg Training Set Accuracy: %f\n', accuracyTot/10 * 100);
%
% missing/missing

input_layer_size=13;
accuracy_1 = zeros(trials,feats);
for j=1:feats,
for i=1:trials,
%%%
%%%	data  = data_raw(randperm(m),:);
%%%	data_train = data(1:540, :);
%%%	data_cv = data(541:600, :);
%%%	data_test = data(601:699, :);
%%%
data  = data_raw(randperm(m),:);
[X mu sig] = featureNormalize(data(:, 3:end));
y=data(:,1);

X_train = X(1:.8*m, :);
X_cv = X(.8*m+1:.9*m, :);
X_test = X(.9*m+1:end, :);

y_train = y(1:.8*m, :);
y_cv = y(.8*m+1:.9*m, :);
y_test = y(.9*m+1:end, :);

%%%	m_train = size(data_train, 1);
%%%	m_cv = size(data_cv, 1);
%%%	m_test = size(data_test, 1);
%%%	
%%%	%The full Data
%%%	X_train = data_train(:, 2:size(data, 2)-1);
%%%	y_train = data_train(:, size(data, 2):end)==4;
%%%
	if j==1,
		X_train_1 = [];
		X_train_2 = X_train(:,2:end);
		y_train_1 = [];
		y_train_2 = y_train;
	elseif j==feats,
		X_train_1 = [];
		X_train_2 = X_train(:,1:end-1);
		y_train_1 = [];
		y_train_2 = y_train;
	else
		X_train_1 = [];
		X_train_2 = [X_train(:,1:j-1) X_train(:,j+1:end)];
		y_train_1 = [];
		y_train_2 = y_train;
	end
	
%%%	X_cv = data_cv(:, 2:size(data, 2)-1);
%%%	y_cv = data_cv(:, size(data, 2):end)==4;
%%%	
%%%	X_test = data_test(:, 2:size(data, 2)-1);
%%%	y_test = data_test(:, size(data, 2):end)==4;
%%%
	if j==1,
		X_test = X_test(:, 2:end);
	elseif j==feats,
		X_test = X_test(:, 1:end-1);
	else
		X_test=[X_test(:, 1:j-1) X_test(:, j+1:end)];
	end
%%%
%%
%%
%%%
%%%
	[nn_params, cost] = trainLinearReg(X_train_1, y_train_1, X_train_2, y_train_2, lambda,input_layer_size, hidden_layer_size, output_layer_size);


	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 	hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 	output_layer_size, (hidden_layer_size + 1));

	pred = predict(Theta1, Theta2, X_test,0);

	acc = mean(double(pred == y_test))
	accuracy_1(i,j) = acc;
end

j
%fprintf('\nTraining Set Accuracy: %f\n', accuracy_1);
end
%%
pause;

%Full/Full

%accuracy = 0;
%for i=1:10,
%
%	data  = data_raw(randperm(m),:);
%	data_train = data(1:540, :);
%	data_cv = data(541:600, :);
%	data_test = data(601:699, :);
%	
%	m_train = size(data_train, 1);
%	m_cv = size(data_cv, 1);
%	m_test = size(data_test, 1);
%	
%	%The full Data
%	X_train = data_train(:, 2:size(data, 2)-1);
%	y_train = data_train(:, size(data, 2):end)==4;
%
%	X_train_1 = X_train(1:m_1,2:end);
%	X_train_2 = [X_train(m_1+1:end,2:end) X_train(m_1+1:end,1)];
%	y_train_1 = y_train(1:m_1);
%	y_train_2 = y_train(m_1+1:end);
%	
%	X_cv = data_cv(:, 2:size(data, 2)-1);
%	y_cv = data_cv(:, size(data, 2):end)==4;
%	
%	X_test = data_test(:, 2:size(data, 2)-1);
%	y_test = data_test(:, size(data, 2):end)==4;
%
%	%X_test = [X_test(:,1:j-1) X_test(:,j+1:end)];
%	X_test = X_test(:, 2:end);
%
%
%
%	[nn_params, cost] = trainLinearReg(X_train_1, y_train_1, X_train_2, y_train_2, lambda,input_layer_size, hidden_layer_size, output_layer_size);
%
%
%	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                 	hidden_layer_size, (input_layer_size + 1));
%
%	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                 	output_layer_size, (hidden_layer_size + 1));
%
%	pred = predict(Theta1, Theta2, X_test,9);
%
%	accuracy = accuracy + mean(double(pred == y_test));
%end
%
%
%fprintf('\nTraining Set Accuracy: %f\n', accuracy/10 * 100);
%
%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%50-50/Missing
%trials=5
%input_layer_size=14;
%accuracy_2 = zeros(trials, feats);
%for j=1:feats,
%for i=1:trials,
%
%data  = data_raw(randperm(m),:);
%[X mu sig] = featureNormalize(data(:, 3:end));
%y=data(:,1);
%
%X_train = X(1:.8*m, :);
%X_cv = X(.8*m+1:.9*m, :);
%X_test = X(.9*m+1:end, :);
%
%y_train = y(1:.8*m, :);
%y_cv = y(.8*m+1:.9*m, :);
%y_test = y(.9*m+1:end, :);
%%
%%	data  = data_raw(randperm(m),:);
%%	data_train = data(1:540, :);
%%	data_cv = data(541:600, :);
%%	data_test = data(601:699, :);
%%	
%%	m_train = size(data_train, 1);
%%	m_cv = size(data_cv, 1);
%%	m_test = size(data_test, 1);
%%	
%%	%The full Data
%%	X_train = data_train(:, 2:size(data, 2)-1);
%%	y_train = data_train(:, size(data, 2):end)==4;
%%	
%%	X_cv = data_cv(:, 2:size(data, 2)-1);
%%	y_cv = data_cv(:, size(data, 2):end)==4;
%%	
%%	X_test = data_test(:, 2:size(data, 2)-1);
%%	y_test = data_test(:, size(data, 2):end)==4;
%%
%%	%X_test = [X_test(:,1:j-1) X_test(:,j+1:end)];
%	if j==1,
%		X_train_1 = X_train(1:m_1_train,2:end);
%		X_train_2 = [X_train(m_1_train+1:end,2:end) X_train(m_1_train+1:end,1)];
%		y_train_1 = y_train(1:m_1_train);
%		y_train_2 = y_train(m_1_train+1:end);
%		X_test = X_test(:, 2:end);
%	elseif j==feats,
%		X_train_1 = X_train(1:m_1_train,1:end-1);
%		X_train_2 = X_train(m_1_train+1:end,:);
%		y_train_1 = y_train(1:m_1_train);
%		y_train_2 = y_train(m_1_train+1:end);
%		X_test = X_test(:, 1:end-1);
%	else
%		X_train_1 = [X_train(1:m_1_train,1:j-1) X_train(1:m_1_train,j+1:end)];
%		X_train_2 = [X_train(m_1_train+1:end,1:j-1) X_train(m_1_train+1:end,j+1:end) X_train(m_1_train+1:end,j)];
%		y_train_1 = y_train(1:m_1_train);
%		y_train_2 = y_train(m_1_train+1:end);
%		X_test=[X_test(:, 1:j-1) X_test(:, j+1:end)];
%	end
%		
%
%
%	[nn_params, cost] = trainLinearReg(X_train_1, y_train_1, X_train_2, y_train_2, lambda,input_layer_size, hidden_layer_size, output_layer_size);
%
%
%	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                 	hidden_layer_size, (input_layer_size + 1));
%
%	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                 	output_layer_size, (hidden_layer_size + 1));
%
%	pred = predict(Theta1, Theta2, X_test,9);
%
%	[pred y_test]
%
%	acc = mean(double(pred == y_test))
%	accuracy_2(i,j) = acc;
%end
%
%j
%
%end
%%accuracy_1;
%%fprintf('\n Means: %f\n', mean(accuracy_1));
%%fprintf('\n Std Devs: %f\n', std(accuracy_1));
%
%
%accuracy_2
%mean(accuracy_2)
%std(accuracy_2)




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%50-50/Full
trials=50;
input_layer_size=14;
hidden_layer_size = 80;
accuracy_3 = zeros(trials, feats);
for j=1:feats,
for i=1:trials,

data  = data_raw(randperm(m),:);
[X mu sig] = featureNormalize(data(:, 3:end));
y=data(:,1);

X_train = X(1:.8*m, :);
X_cv = X(.8*m+1:.9*m, :);
X_test = X(.9*m+1:end, :);

y_train = y(1:.8*m, :);
y_cv = y(.8*m+1:.9*m, :);
y_test = y(.9*m+1:end, :);



%
%	data  = data_raw(randperm(m),:);
%	data_train = data(1:540, :);
%	data_cv = data(541:600, :);
%	data_test = data(601:699, :);
%	
%	m_train = size(data_train, 1);
%	m_cv = size(data_cv, 1);
%	m_test = size(data_test, 1);
%	
%	%The full Data
%	X_train = data_train(:, 2:size(data, 2)-1);
%	y_train = data_train(:, size(data, 2):end)==4;
%	
%	X_cv = data_cv(:, 2:size(data, 2)-1);
%	y_cv = data_cv(:, size(data, 2):end)==4;
%	
%	X_test = data_test(:, 2:size(data, 2)-1);
%	y_test = data_test(:, size(data, 2):end)==4;
%
%	%X_test = [X_test(:,1:j-1) X_test(:,j+1:end)];
	if j==1,
		X_train_1 = X_train(1:m_1_train,2:end);
		X_train_2 = [X_train(m_1_train+1:end,2:end) X_train(m_1_train+1:end,1)];
		y_train_1 = y_train(1:m_1_train);
		y_train_2 = y_train(m_1_train+1:end);

		X_cv_1 = X_cv(1:m_1_cv,2:end);
		X_cv_2 = [X_cv(m_1_cv+1:end,2:end) X_cv(m_1_cv+1:end,1)];
		y_cv_1 = y_cv(1:m_1_cv);
		y_cv_2 = y_cv(m_1_cv+1:end);
		X_test = [X_test(:, 2:end) X_test(:, 1)];
	elseif j==feats,
		X_train_1 = X_train(1:m_1_train,1:end-1);
		X_train_2 = X_train(m_1_train+1:end,:);
		y_train_1 = y_train(1:m_1_train);
		y_train_2 = y_train(m_1_train+1:end);

		X_cv_1 = X_cv(1:m_1_cv,1:end-1);
		X_cv_2 = X_cv(m_1_cv+1:end,:);
		y_cv_1 = y_cv(1:m_1_cv);
		y_cv_2 = y_cv(m_1_cv+1:end);
		%X_test = X_test(:, 1:end-1);
	else
		X_train_1 = [X_train(1:m_1_train,1:j-1) X_train(1:m_1_train,j+1:end)];
		X_train_2 = [X_train(m_1_train+1:end,1:j-1) X_train(m_1_train+1:end,j+1:end) X_train(m_1_train+1:end,j)];
		y_train_1 = y_train(1:m_1_train);
		y_train_2 = y_train(m_1_train+1:end);

		X_cv_1 = [X_cv(1:m_1_cv,1:j-1) X_cv(1:m_1_cv,j+1:end)];
		X_cv_2 = [X_cv(m_1_cv+1:end,1:j-1) X_cv(m_1_cv+1:end,j+1:end) X_cv(m_1_cv+1:end,j)];
		y_cv_1 = y_cv(1:m_1_cv);
		y_cv_2 = y_cv(m_1_cv+1:end);
		X_test=[X_test(:, 1:j-1) X_test(:, j+1:end) X_test(:,j)];
	end

%		
%
%
%
	[nn_params, cost] = trainLinearReg(X_train_1, y_train_1, X_train_2, y_train_2, lambda,input_layer_size, hidden_layer_size, output_layer_size);


	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 	hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 	output_layer_size, (hidden_layer_size + 1));

	pred = predict(Theta1, Theta2, X_test,0);

	acc = mean(double(pred == y_test))

%	[pred y_test]

	accuracy_3(i,j) = acc;
end

j
%[lambda_vec, error_train, error_val] = ...
%    validationCurve( X_train_1, y_train_1, X_train_2, y_train_2, X_cv_1, y_cv_1,X_cv_2, y_cv_2,input_layer_size, hidden_layer_size, output_layer_size);
%
%[lambda_vec error_train error_val]

end


accuracy_3
mean(accuracy_3)
std(accuracy_3)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%50 (Full)/Full
%for j=1:9,
%accuracy = 0;
%for i=1:10,
%
%	data  = data_raw(randperm(m),:);
%	data_train = data(1:540, :);
%	data_cv = data(541:600, :);
%	data_test = data(601:699, :);
%	
%	m_train = size(data_train, 1);
%	m_cv = size(data_cv, 1);
%	m_test = size(data_test, 1);
%	
%	%The full Data
%	X_train = data_train(1:m_1, 2:size(data, 2)-1);
%	y_train = data_train(1:m_1, size(data, 2):end)==4;
%	
%	X_cv = data_cv(:, 2:size(data, 2)-1);
%	y_cv = data_cv(:, size(data, 2):end)==4;
%	
%	X_test = data_test(:, 2:size(data, 2)-1);
%	y_test = data_test(:, size(data, 2):end)==4;
%
%		
%
%
%
%	[nn_params, cost] = trainLinearReg([], [], X_train, y_train, lambda,input_layer_size, hidden_layer_size, output_layer_size);
%
%
%	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                 	hidden_layer_size, (input_layer_size + 1));
%
%	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                 	output_layer_size, (hidden_layer_size + 1));
%
%	pred = predict(Theta1, Theta2, X_test,0);
%
%	accuracy = accuracy + mean(double(pred == y_test));
%end
%
%j
%fprintf('\nTraining Set Accuracy: %f\n', accuracy/10 * 100);
%end

