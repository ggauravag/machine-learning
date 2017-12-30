%% Program to apply machine learning on give wine quality data

%%

clear; close all; clc
fprintf('### Loading training data for wine quality ###\n\n');

filename = input('Enter the filename to be loaded: ', 's');
loaded_data = dlmread(filename, ';', 1, 0);
dataSize = size(loaded_data);
printf("Size of data: %d %d\n", dataSize(1), dataSize(2));

X = loaded_data(:, 1:dataSize(2) - 1);
y = loaded_data(:, dataSize(2):dataSize(2));
[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(dataSize(1), 1) X];

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(dataSize(2), 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');
totalCost = 0;
for i = 1:dataSize(1),
  totalCost += ((X(i, :) * theta) - y(i)) ^ 2;
  %printf("Actual: %f, Computed: %f\n", y(i), X(i, :) * theta);
end

printf('Total Cost: %f\n', totalCost);

%
% USING NORMAL EQUATION FOR THE RESULTS
%

X = loaded_data(:, 1:dataSize(2) - 1);
X = [ones(dataSize(1), 1) X];
y = loaded_data(:, dataSize(2):dataSize(2));
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

totalCost = 0;
for i = 1:dataSize(1),
  totalCost += ((X(i, :) * theta) - y(i)) ^ 2;
  %printf("Actual: %f, Computed: %f\n", y(i), X(i, :) * theta);
end

printf('Total Cost: %f\n', totalCost);
