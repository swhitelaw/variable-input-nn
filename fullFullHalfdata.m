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


m = size(data_raw, 1)
m_1 = floor(m/2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Leaf Lambda
%lambda=0.03;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Cancer Lanbda
%lambda=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Glass Lanbda
lambda=0.1;

%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%50 (Full)/Full
accuracy_4 = zeros(trials, feats);
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


X_train = X(1:floor(.4*m), :);
X_cv = X(floor(.4*m)+1:floor(.45*m), :);
X_test = X(floor(.45*m)+1:floor(.5*m), :);

y_train = y(1:floor(.4*m), :);
y_cv = y(floor(.4*m)+1:floor(.45*m), :);
y_test = y(floor(.45*m)+1:floor(.5*m), :);



		



	[nn_params, cost] = trainLinearReg([], [], X_train, y_train, lambda,input_layer_size, hidden_layer_size, output_layer_size);


	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 	hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 	output_layer_size, (hidden_layer_size + 1));

	pred = predict(Theta1, Theta2, X_test,0);

	accuracy_4(i,j) = mean(double(pred == y_test));
end

j
end

