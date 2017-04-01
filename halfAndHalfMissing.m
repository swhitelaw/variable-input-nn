%% Initialization
clear ; close all; clc
setenv('GNUTERM','qt')
%% Setup the parameters you will use for this exercise
input_layer_size  = 9;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
output_layer_size = 7;         % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
feats = 9;		%number of features
trials = 100;	%number of trials
%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Leaf Data
%data_raw = csvread('leaf.csv');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Cancer Data
%data_raw = csvread('data.csv');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Glass Data
data_raw = csvread('glass.csv');

m = size(data_raw, 1);
m_1 = floor(m/2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Leaf Lambda
%lambda=0.03;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Cancer Lanbda
%lambda=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Glass Lanbda
lambda=0.1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%50-50/Missing
accuracy_2 = zeros(trials, feats);
for j=1:feats,
for i=1:trials,

data  = data_raw(randperm(m),:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Leaf Processing
%[X mu sig] = featureNormalize(data(:, 3:end));
%y=data(:,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Cancer Processing
%[X mu sig] = featureNormalize(data(:, 2:end-1));
%y=data(:,size(data,2))./2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Glass Processing
[X mu sig] = featureNormalize(data(:, 2:end-1));
y=data(:,size(data,2));


X_train = X(1:floor(.8*m), :);
X_cv = X(floor(.8*m)+1:floor(.9*m), :);
X_test = X(floor(.9*m)+1:end, :);

m_train = size(X_train, 1);
m_1_train = floor(m_train/2);

m_cv = size(X_cv, 1);
m_1_cv = floor(m_cv/2);

y_train = y(1:floor(.8*m), :);
y_cv = y(floor(.8*m)+1:floor(.9*m), :);
y_test = y(floor(.9*m)+1:end, :);


	if j==1,
		X_train_1 = X_train(1:m_1_train,2:end);
		X_train_2 = [X_train(m_1_train+1:end,2:end) X_train(m_1_train+1:end,1)];
		y_train_1 = y_train(1:m_1_train);
		y_train_2 = y_train(m_1_train+1:end);
		X_test = X_test(:, 2:end);
	elseif j==feats,
		X_train_1 = X_train(1:m_1_train,1:end-1);
		X_train_2 = X_train(m_1_train+1:end,:);
		y_train_1 = y_train(1:m_1_train);
		y_train_2 = y_train(m_1_train+1:end);
		X_test = X_test(:, 1:end-1);
	else
		X_train_1 = [X_train(1:m_1_train,1:j-1) X_train(1:m_1_train,j+1:end)];
		X_train_2 = [X_train(m_1_train+1:end,1:j-1) X_train(m_1_train+1:end,j+1:end) X_train(m_1_train+1:end,j)];
		y_train_1 = y_train(1:m_1_train);
		y_train_2 = y_train(m_1_train+1:end);
		X_test=[X_test(:, 1:j-1) X_test(:, j+1:end)];
	end
		


	[nn_params, cost] = trainLinearReg(X_train_1, y_train_1, X_train_2, y_train_2, lambda,input_layer_size, hidden_layer_size, output_layer_size);


	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 	hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 	output_layer_size, (hidden_layer_size + 1));

	pred = predict(Theta1, Theta2, X_test,9);


	acc = mean(double(pred == y_test))
	accuracy_2(i,j) = acc;
end

j

end
%accuracy_1;
%fprintf('\n Means: %f\n', mean(accuracy_1));
%fprintf('\n Std Devs: %f\n', std(accuracy_1));


accuracy_2
mean(accuracy_2)
std(accuracy_2)
