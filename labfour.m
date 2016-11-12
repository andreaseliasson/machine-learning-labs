% Load Boston Housing Data from UCI ML Repository
load -ascii housing.data;

% Normalize the data, zero mean, unit standard deviation
[N, p1] = size(housing);
p = p1-1;
Y = [housing(:,1:p) ones(N,1)];
for j=1:p
    Y(:,j)=Y(:,j)-mean(Y(:,j));
    Y(:,j)=Y(:,j)/std(Y(:,j));
end

% Again, normalize, zero mean, unit standard deviation
f = housing(:,p1);
f = f - mean(f);
f = f/std(f);

% You can predict the response variable (output variable) f, the house price, from the
% covariates (input variable) by estimating a linear regression:

% Least squares regression as pseudo inverse

w = inv(Y' * Y) * Y' * f;
fh = Y * w;

figure(1), clf,
plot(f, fh, 'r.', 'LineWidth', 2),
grid on;
s=getenv('USERNAME');
xlabel('True House Price', 'FontSize', 14);
ylabel('Prediction', 'FontSize', 14);
title(['Linear Regression: ' s], 'FontSize', 14);

% Split the data into a training set and a test set, estimate the regression model (w) on the
% training set and see how training and test errors differ.

training_set = Y(1:N/2, :);
test_set = Y(N/2+1:N, :);

% Train on data (training set)
params_test_set = inv(training_set' * training_set) * training_set' * f(1:N/2, :);

% Predict on seen data
fh_training_set = training_set * params_test_set;
% Predict on unseen data
fh_test_set = test_set * params_test_set;

% Seen data
figure(2), clf,
plot(f(1:N/2, :), fh_training_set, 'b.', 'LineWidth', 2),
grid on;
xlabel('True House Price', 'FontSize', 14);
ylabel('Prediction (seen data)', 'FontSize', 14);
title(['Linear Regression: Seen data' s], 'FontSize', 14);

% Unseen data
figure(3), clf,
plot(f(N/2+1:N, :), fh_test_set, 'g.', 'LineWidth', 2),
grid on;
xlabel('True House Price', 'FontSize', 14);
ylabel('Prediction (unseen data)', 'FontSize', 14);
title(['Linear Regression: Unseen data' s], 'FontSize', 14);

% Implement 10-fold cross validation on the data and quantify an average prediction error
% and an uncertainty on it.

indices = crossvalind('Kfold', N, 10);
test_set1 = (indices == 1);
train_set1 = ~test_set1;
% for jj = 1:10
%    test_set = (indices == i);
%    train_set = ~test_set;
%    
% end
