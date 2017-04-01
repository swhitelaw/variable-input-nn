%% Initialization
clear ; close all; clc
setenv('GNUTERM','qt')
%% Setup the parameters you will use for this exercise
input_layer_size  = 9;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
output_layer_size = 7;         % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)


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


for i=1:10,

error_tot=zeros(7);

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

%% find lambda Training with Full data
[lambda_vec, error_train, error_val] = ...
    validationCurve( [], [], X_train, y_train, [], [],X_cv, y_cv,input_layer_size, hidden_layer_size, output_layer_size);
%%
%close all;
%plot(lambda_vec, error_train, lambda_vec, error_val);
%legend('Train', 'Cross Validation');
%xlabel('lambda');
%ylabel('Error');

lambda_vec
error_val

error_tot=error_tot+error_val;

end
error_tot./10

