% 1 - load data, normalize, and partitions into training and test sets with
% corresponding outputs (targets)

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

% Split into training and test sets with corresponding target vectors

Xtr = Y(1:N/2, :);
ytr = f(1:N/2, :);
Ntr = N/2;

Xts = Y(N/2+1:N, :);
yts = f(N/2+1:N, :);

% 2 Set the widths of the basis functions to a sensible scale
sig = norm(Xtr(ceil(rand*Ntr),:) - Xtr(ceil(rand*Ntr),:));

% 3 Perform K-means clustering to find centres for the basis functions. Use
% K = Ntr / 10
K = round(Ntr/10);
% K = 100;
[idx, C] = kmeans(Xtr, K);

% 4 Construct the design matrix
A = ones(Ntr, p);
A = [A ones(Ntr, 1)];

for i=1:Ntr
     for j=1:K
        A(i, j) = exp(-norm(Xtr(i,:) - C(j,:)) / sig^2);
     end
end

% 5 Solve for the weights (parameters)
lambda = A \ ytr;

% 7 What does the model predict at each of the training data?
yh = zeros(Ntr, 1);
u = zeros(1, K);

for n=1:Ntr
   for j=1:K
      u(j) = exp(-norm(Xtr(n,:) - C(j,:)) / sig^2); 
   end
   yh(n) = u(1,:) * lambda;
end

figure(1), clf,
plot(ytr, yh, 'rx', 'LineWidth', 2), grid on,
title('RBF Prediction on Training Data', 'FontSize', 16);
axis([-2 3 -1 3]);
xlabel('Target', 'FontSize', 14);
ylabel('Prediction', 'FontSize', 14);

print -depsc f5-1.eps;

% 7 What does the model predict at each point in the test set?
yhUnSeen = zeros(Ntr, 1);
uts = zeros(1, K);

for n=1:Ntr
   for j=1:K
      uts(j) = exp(-norm(Xts(n,:) - C(j,:)) / sig^2);
   end
   yhUnSeen(n) = uts(1,:) * lambda;
end

figure(2), clf,
plot(yts, yhUnSeen, 'bx', 'LineWidth', 2), grid on,
title('RBF Prediction on Unseen Data', 'FontSize', 16);
axis([-2 3 -1 3]);
xlabel('Target', 'FontSize', 14);
ylabel('Prediction', 'FontSize', 14);

print -depsc f5-2.eps;

% How do the training and test errors compare?
% training set errors
tr_errors = zeros(Ntr, 1);

for iter1=1:Ntr
    tr_errors(iter1) = abs(ytr(iter1) - yh(iter1));
end

tr_errors_mean = mean(tr_errors);

% test set errors
ts_errors = zeros(Ntr, 1);

for iter1=1:Ntr
    ts_errors(iter1) = abs(yts(iter1) - yhUnSeen(iter1));
end

ts_errors_mean = mean(ts_errors);

ks = 15:1:115;
tr_prediction_errors_mean2 = zeros(101, 1);
ts_prediction_errors_mean2 = zeros(101, 1);

for iter2=1:101
    [idx2, C2] = kmeans(Xtr, ks(iter2));
    
    A2 = ones(Ntr, ks(iter2));
    for i=1:Ntr
         for j=1:ks(iter2)
            A2(i, j) = exp(-norm(Xtr(i,:) - C2(j,:)) / sig^2);
         end
    end
    
    % Solve for the paramaters (weights) using pseudo inverse
    lambda2 = A2 \ ytr;
    
    % Evaluate test set prediction
    B2 = ones(Ntr, ks(iter2));

    for i=1:Ntr
         for j=1:ks(iter2)
            B2(i, j) = exp(-norm(Xts(i,:) - C2(j,:)) / sig^2);
         end
    end
    
    tr_predictions2 = A2 * lambda2;
    ts_predictions2 = B2 * lambda2;
    
    tr_prediction_errors2 = zeros(Ntr, 1);
    ts_prediction_errors2 = zeros(Ntr, 1);
    
    for iter33=1:Ntr
       tr_prediction_errors2(iter33) = abs(ytr(iter33) - tr_predictions2(iter33));
       ts_prediction_errors2(iter33) = abs(yts(iter33) - ts_predictions2(iter33));
    end
    
    tr_prediction_errors_mean2(iter2) = mean(tr_prediction_errors2);
    ts_prediction_errors_mean2(iter2) = mean(ts_prediction_errors2);
end

% Plot K vs training prediction errors
figure(3),clf,
plot(ks, tr_prediction_errors_mean2, 'b', 'LineWidth', 2),
title('K vs training set prediction errors', 'FontSize', 16);
% axis([0 120 0 0.3]);
xlabel('Basis functions', 'FontSize', 14);
ylabel('Mean Prediction Error', 'FontSize', 14);

print -depsc f5-3.eps;

% Plot K vs test prediction errors
figure(4),clf,
plot(ks, ts_prediction_errors_mean2, 'b', 'LineWidth', 2),
title('K vs test set prediction errors', 'FontSize', 16);
% axis([0 120 0 0.6]);
xlabel('Basis functions', 'FontSize', 14);
ylabel('Mean Prediction Error', 'FontSize', 14);

print -depsc f5-4.eps;

% 8 Compare your results with the linerar regression model for lab 4. Does
% the use of a nonlinear model improve predictions?

rbf_prediction_error_means = zeros(20, 1);
lg_prediction_error_means = zeros(20, 1);

for iter=1:20
    % Partition into random training and test sets
    ii = randperm(N);
    Xtr1 = Y(ii(1:N/2), :);
    ytr1 = f(ii(1:N/2), :);

    Xts1 = Y(ii(N/2+1:N), :);
    yts1 = f(ii(N/2+1:N), :);
    
    % Train on training set
    % Construct design matrix

    A1 = ones(Ntr, K);

    for i=1:Ntr
         for j=1:K
            A1(i, j) = exp(-norm(Xtr1(i,:) - C(j,:)) / sig^2);
         end
    end
    
    % Solve for the paramaters (weights) using pseudo inverse
    lambda1 = A1 \ ytr1;
    
    % Evaluate test set prediction
    B1 = ones(Ntr, K);

    for i=1:Ntr
         for j=1:K
            B1(i, j) = exp(-norm(Xts1(i,:) - C(j,:)) / sig^2);
         end
    end

    test_set_predictions = B1 * lambda1;
    
    test_set_prediction_errors = zeros(Ntr, 1);
    
    for jj=1:Ntr
        test_set_prediction_errors(jj) = abs(yts1(jj) - test_set_predictions(jj));
    end

    rbf_prediction_error_means(iter) = mean(test_set_prediction_errors);
    
    % Linear regression model
    
    % solve for paramaters (weights)
    
    lg_params = Xtr1 \ ytr1;
    
    % fit model on test set
    
    lg_predictions = Xts1 * lg_params;
    lg_prediction_errors = zeros(Ntr, 1);
    
    for jj2=1:Ntr
       lg_prediction_errors(jj2) = abs(yts1(jj2) - lg_predictions(jj2)); 
    end
    
    lg_prediction_error_means(iter) = mean(lg_prediction_errors);
end

% Plot boxplots
figure(5),
boxplot([lg_prediction_error_means rbf_prediction_error_means], 'Labels', {'Linear', 'RBF'}),
ylabel('Mean Squared Difference prediction error', 'FontSize', 14);
title('Prediction errors: Linear vs RBF', 'FontSize', 14);

print -depsc f5-5.eps;