%1
% Plot line with points from linear definition 3x + 4y + 12 = 0
hold off;
plot([0 -3], [-4 0], 'b', 'LineWidth', 2);
axis([-5 5 -5 5]);
grid on;

print -depsc f2-1.eps;

%Perpinducalar distance from the origin to the straight line above
pDistance = (-12) / sqrt(3^2 + 4^2);
% => -2.4000

%2
% distinct means for each bi-variate Gaussian density
m1 = [0 2];
m2 = [1.5 0.0];

%Identical covariance matrix
C = [2 1; 1 2];

% Generate two bi-variate Gaussian densities
N = 100;
X = randn(N, 2);
Y = randn(N, 2);

% Use cholesky transformation
X = X * chol(C);
Y = Y * chol(C);

% Shift the mean for each of the columns in the two bi-variate Gaussian
% densities
X1 = X + kron(ones(N, 1), m1);
Y1 = Y + kron(ones(N, 1), m2);

plot(X1(:, 1), X1(:, 2), 'o', Y1(:, 1), Y1(:, 2), 'mx');

print -depsc f2-2.eps;

%3
% Compute the Bayes' optimal class boundary
% Linear classifier denoted by w^t x + b <> 0
w = 2 * C^-1 * transp((m1 - m2));
b = (m1 * C^-1 * transp(m1) - m2 * C^-1 * transp(m2));

xIntercept = -b / w(2, 1);
yIntercept = -b / w(1, 1);

slope = (yIntercept - 0.00) / (0.00 - xIntercept);

% Using y = mx + b for definting new x and y points on the line
xCoord1 = -3.00;
yCoord1 = slope * xCoord1 + yIntercept;

xCoord2 = 4.00;
yCoord2 = slope * xCoord2 + yIntercept;


hold on;
% plot([xIntercept, 0], [0, yIntercept], 'b', 'LineWidth', 2);
plot([xCoord1, xCoord2], [yCoord1, yCoord2], 'b', 'LineWidth', 2);

print -depsc f2-3.eps;

% 4 The Perceptron Learning Algorithm
% X = N by 2 matrix of data
% Place the two by-matrices all in one N by 2 matrix
X2 = [X1', Y1']';
% y classlabels -1 and +1
yClassLabels = [-ones(1, 100), ones(1, 100)];

N1 = 200;
% Include columns of ones for bias term (X3 is our accumulated input matrix of examples)
% X2 is the combination of the two bi-variate Gaussian distributions
X3 = [X2 ones(N1, 1)];

% Separate into training and test sets
ii = randperm(N1);
Xtr = X3(ii(1:N1/2), :);
ytr = yClassLabels(:, ii(1:N1/2));
Xts = X3(ii(N1/2+1: N1), :);
yts = yClassLabels(:, ii(N1/2+1: N1));

% Initialize weights (This is the weight vector we are trying to learn)
w = randn(3, 1);

% Error correcting learning
eta = 0.001;
for iter=1:5000
    j = ceil(rand * N1/2);
    if ( ytr(j) * Xtr(j, :) * w < 0 )
        w = w + (eta * Xtr(j, :)' * ytr(j));
    end
end

% Performance on test data
yhts = Xts * w;
disp([yts' yhts]);

errors = size(find(yts' .* yhts < 0));
percentageErrorRate = 100 * errors(1) / (N1/2);

xInterceptP = -w(3, 1) / w(2, 1);
yInterceptP = -w(3, 1) / w(1, 1);

slopeP = (yInterceptP - 0.00) / (0.00 - xInterceptP);

xCoordP1 = -3.00;
yCoord1 = slopeP * xCoordP1 + yIntercept;

xCoordP2 = 4.00;
yCoordP2 = slopeP * xCoordP2 + yIntercept;

hold on;
% plot([xInterceptP 0], [0 yInterceptP], 'g', 'LineWidth', 2);
plot([xCoordP1 xCoordP2], [yCoord1 yCoordP2], 'g', 'LineWidth', 2);

print -depsc f2-4.eps;