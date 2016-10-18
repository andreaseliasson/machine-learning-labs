%2
x = rand(1000, 1);
hist(x, 40);
[xx, nn] = hist(x);

x = randn(1000, 1);

N = 1000;
x1 = zeros(N, 1);
% Central limit theorem doing its work
% Taking the sum (or average) of the differences between any two numbers
% for a repeated number of times (in this case 1000)
% will result in a Gaussian Distribution.
% This explains the shift from a uniform distribution to a Gaussian
% distribution.
for n=1:N
    x1(n, 1) = sum(rand(12, 1) - rand(12, 1));
end
hist(x1, 40);

print -depsc f01.eps;

%3
C = [2 1; 1 2];
A = chol(C);
X = randn(1000, 2);
% Transform each of the two dimensional vectors (rows of X)
Y = X * A;
plot(X(:,1), X(:,2), 'c.', Y(:,1), Y(:,2), 'mx');

print -depsc f02.eps;
% Both of the X variables and Y variables seem to have a linear
% relationship
% Transorming a Gaussian distribution by a linear matrix will result in
% another Gaussian distribution.

theta = 0.25;
% What does this line do?
% Projects the data in Y in the direction of [sin(theta() cos(theta)] <--
% this is the vector to which we project on to.
yp = Y * [sin(theta); cos(theta)];
answer = var(yp);

N = 50;
plotArray = zeros(N, 1);
thetaRange = linspace(0, 2*pi, N);

for n=1:N
    theta = thetaRange(:, n);
    % Here we are projecting each row of Y (vector) on to the [sin(theta); cos(theta)]
    % Basically, we are transforming a vector in a 2d space to a vector in
    % a one dimensional space.
    scalarProjection = Y * [sin(theta); cos(theta)];
    plotArray(n, :) = var(scalarProjection);
end

plot(plotArray);

print -depsc f03.eps;


%Explain what you observe by calculating the eigenvectors of the covariance
%matrix.

% As we project the data in the Y variable on to the vector defined by the
% sine and cosine we see how the variance changes for each sine and cosine
% vector.

% print -depsc f1.eps

