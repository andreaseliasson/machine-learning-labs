load -ascii yacht_hydro.data;

% Normalize the data, zero mean, unit standard deviation
[N, p1] = size(yacht_hydro);
p = p1-1;
Y = [yacht_hydro(:,1:p) ones(N,1)];
for j=1:p
    Y(:,j)=Y(:,j)-mean(Y(:,j));
    Y(:,j)=Y(:,j)/std(Y(:,j));
end

% Again, normalize, zero mean, unit standard deviation
f = yacht_hydro(:,p1);
f = f - mean(f);
f = f/std(f);

% Iteratively transform our training set using a non-linear model, Gaussian RBF, and
% solve for our unknown parameters, i.e. minimimizing our error function,
% using the normal equation (psedo-inverse)

% Construct design matrix
rbf_tr_prediction_error_means = zeros(20, 1);
rbf_ts_prediction_error_means = zeros(20, 1);

lg_tr_prediction_error_means = zeros(20, 1);
lg_ts_prediction_error_means = zeros(20, 1);

for i=1:20
    % Gaussian RBF model
    
    % Partition into random training and test sets
    ii = randperm(N);
    Xtr = Y(ii(1:N/2), :);
    ytr = f(ii(1:N/2), :);
    Ntr = N/2;

    Xts = Y(ii(N/2+1:N), :);
    yts = f(ii(N/2+1:N), :);
    
    % Set sigma and centers to fixed values
    sig = norm(Xtr(ceil(rand*Ntr),:) - Xtr(ceil(rand*Ntr),:));

    % Use K-means clustering to get K number of centers
    K = round(Ntr/10);
    [idx, C] = kmeans(Xtr, K);
    
    A = ones(Ntr, K);

    for ii=1:Ntr
        for j=1:K
            A(ii, j) = exp(-norm(Xtr(ii,:) - C(j,:)) / sig^2);
        end
    end

    % Minimize error function to solve for our unknown parameters
    lambda = A \ ytr;

    % How does it perform on training data?
    yPredictionsTraining = A * lambda;
    tr_prediction_errors = zeros(Ntr, 1);

    for iii=1:Ntr
        tr_prediction_errors = abs(ytr(iii) - yPredictionsTraining(iii));
    end
    rbf_tr_prediction_error_means(i) = mean(tr_prediction_errors);
    
    % How does it perform on test data?
    
    % Construct our design matrix
    B = ones(Ntr, K);

    for ii=1:Ntr
        for j=1:K
            B(ii, j) = exp(-norm(Xtr(ii,:) - C(j,:)) / sig^2);
        end
    end
    
    ytsPredictionsTraining = B * lambda;
    ts_prediction_errors = zeros(Ntr, 1);

    for iii=1:Ntr
        ts_prediction_errors = abs(yts(iii) - ytsPredictionsTraining(iii));
    end
    rbf_ts_prediction_error_means(i) = mean(ts_prediction_errors);
    
    % Linear model
    
    % Our design matrix is already defined as Xtr
    % Mimize our least sqaured error function using the normal equation (pseudo inverse)
    
    % w = inv(Y' * Y) * Y' * ytr;
    w = Xtr \ ytr;
    
    % How does it perform on seen data?
    lg_ytr = Xtr * w;
    lg_tr_prediction_errors = zeros(Ntr, 1);
    
    for ii=1:Ntr
        lg_tr_prediction_errors = abs(ytr(ii) - lg_ytr(ii));
    end
    lg_tr_prediction_error_means(i) = mean(lg_tr_prediction_errors);
    
    % How does it perform on unseen data?
    lg_yts = Xts * w;
    lg_ts_prediction_errors = zeros(Ntr, 1);
    
    for i1=1:Ntr
        lg_ts_prediction_errors = abs(yts(i1) - lg_yts(i1));
    end
    lg_ts_prediction_error_means(i) = mean(lg_ts_prediction_errors);
end

% Plot boxplots
figure(1),
boxplot([lg_ts_prediction_error_means rbf_ts_prediction_error_means], 'Labels', {'Linear', 'RBF'}),
ylabel('Mean Squared Difference prediction error', 'FontSize', 14);
title('Prediction errors - Linear vs RBF - Yacht Hydrodynamics', 'FontSize', 14);

print -depsc f5-6.eps;

