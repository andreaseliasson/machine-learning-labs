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

predictionErrors = zeros(10, 1);
modulo = mod(N, 10);
group = floor(N/10);

for jj = 1:10
    if (jj == 1)
        testSet = Y(1:group, :);
        trainingSet = Y(group:N, :);
        f10 = f(1:group, :);
        w10 = inv(trainingSet' * trainingSet) * trainingSet' * f(group:N, :);
        fh10 = testSet * w10;
        
        predictionError = zeros(50, 1);
        for jjj = 1:50
            predictionError(jjj, :) = abs(f10(jjj, :) - fh10(jjj, :));
        end
        predictionErrors(jj, :) = mean(predictionError);

    elseif (jj ~= 1 && jj ~= 10)
        testSet = Y(jj * group:jj * group + group, :);
        trainingSet1 = Y(1:jj * group, :);
        trainingSet2 = Y(jj * group + group:N, :);
        trainingSet = [trainingSet1; trainingSet2];
        % Actual values in training data
        ff10 = [f(1:jj * group, :); f(jj * group + group:N, :)];
        % Actual values in test data
        f10 = f(jj * group:jj * group + group, :);
        w10 = inv(trainingSet' * trainingSet) * trainingSet' * ff10;
        fh10 = testSet * w10;
        
        predictionError = zeros(50, 1);
        for jjj = 1:50
            predictionError(jjj, :) = abs(f10(jjj, :) - fh10(jjj, :));
        end
        predictionErrors(jj, :) = mean(predictionError);
        
    else
        testSet = Y((jj - 1) * group: N, :);
        trainingSet = Y(1:(jj - 1) * group, :);
        f10 = f((jj - 1) * group:N, :);
        w10 = inv(trainingSet' * trainingSet) * trainingSet' * f(1:(jj -1) * group);
        fh10 = testSet * w10;
        
        predictionError = zeros(56, 1);
        for jjj = 1:56
            predictionError(jjj, :) = abs(f10(jjj, :) - fh10(jjj, :));
        end
        predictionErrors(jj, :) = mean(predictionError);
    end
end

avgPredictionError = mean(predictionErrors);
standardDeviationError = std(predictionErrors);
figure(4),
boxplot(predictionErrors),
title(['Box plot of prediction errors' s], 'FontSize', 14);

% Regression using the CVX Tool
% The least squares regression we have done above can be implemented as
% follows in the cvx tool:

cvx_begin quiet
    variable w1( p+1 );
    minimize norm( Y*w1 - f );
cvx_end
fh1 = Y * w1;

% Check if the two methods produce the same results
figure(5), clf,
plot(w, w1, 'mx', 'Linewidth', 2);

% It appears as if the two methods produce the same result (parameters in
% our hypothesis function).

% Sparse Regression

% Let us now regularize the regression w2 = minw|Yw - f| + gamma * |w|1. We can
% implement this as follows.

gamma = 8.0;
cvx_begin quiet
    variable w2( p+1 );
    minimize( norm(Y*w2-f) + gamma*norm(w2,1) );
cvx_end

fh2 = Y * w2;

figure(6), clf,
plot(f, fh2, 'b.', 'LineWidth', 2);
grid on;
axis([-2 3 -3 3]);
xlabel('True House Price', 'FontSize', 14);
ylabel('Prediction', 'FontSize', 14);
title('Sparse Regression', 'FontSize', 14);

% You can find the non-zero coefficients (parameters) that are not switched
% off by the regularizer.

[isNzero] = find(abs(w2) > 1e-5);
disp('Relevant variables');
disp(isNzero);

% I seem to only get 3 relevant variables, 6, 11, 13, indicating 
% avg number of rooms per dwelling, per value property tax rate per 10,000
% and lower status of the population, respectively

% Change the paramater gamma over the range 0.01 -> 40 in 100 steps and
% plot a graph of how the number of non-zero coeficients changes with
% increasing regularization.

gammaRange = linspace(0.01, 40, 100);
nonZeroCoefficients = ones(100, 1);
for ii = 1:100
    currentGamma = gammaRange(ii);
    
    cvx_begin quiet
        variable w2G( p+1 );
        minimize( norm(Y*w2G-f) + currentGamma * norm(w2G,1) );
    cvx_end
    
    nonZeros = find(abs(w2G) > 1e-5);
    nonZeroCoefficients(ii, 1) = size(nonZeros, 1);
end

figure(7), clf,
plot(gammaRange, nonZeroCoefficients, 'b', 'Linewidth', 2);
