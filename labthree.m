hold off;
% 1
% Define a two class pattern classification problem in two dimensions with
% the the following mean and covariance parameters.

m1 = [0 2]';
m2 = [1.7 2.5]';

C1 = [2 1; 1 2];
C2 = [2 1; 1 2];

numGrid = 50;
xRange = linspace(-6.0, 6.0, numGrid);
yRange = linspace(-6.0, 6.0, numGrid);
P1 = zeros(numGrid, numGrid);
P2 = P1;


for i=1:numGrid
    for j=1:numGrid
        x = [yRange(j) xRange(i)]';
        P1(i,j) = mvnpdf(x', m1', C1);
        P2(i,j) = mvnpdf(x', m2', C2);
    end
end
Pmax = max(max([P1 P2]));
figure(1), clf, contour(xRange, yRange, P1, [0.1*Pmax 0.5*Pmax 0.8*Pmax], 'LineWidth', 2);
hold on;
plot(m1(1), m1(2), 'b*', 'LineWidth', 4);
contour(xRange, yRange, P2, [0.1*Pmax 0.5*Pmax 0.8*Pmax], 'LineWidth', 2);
plot(m2(1), m2(2), 'r*', 'LineWidth', 4);

print -depsc f3-1.eps;

%2
% Draw 200 samples from each of the two distributions and plot them on the
% top of the countours


N = 200;
X1 = mvnrnd(m1, C1, N);
X2 = mvnrnd(m2, C2, N);

figure(1);
hold on;
plot(X1(:, 1), X1(:, 2), 'bx', X2(:, 1), X2(:, 2), 'ro');
grid on;

print -depsc f3-2.eps;

% 3
% Compute the Fisher Linear Discriminant direction using the means and
% covariance matrcices of the problem, and plot the discriminant direction.

wF = inv(C1+C2)*(m1-m2);
xx = -6:0.1:6;
yy = xx * wF(2)/wF(1);
figure(1);
hold on;
plot(xx,yy, 'r', 'LineWidth', 2);

print -depsc f3-3.eps;

% randomDirection = [-0.6534; 0.1311];
% randomDirection = rand(2, 1);
% xxR = -6:0.1:6;
% yyR = xxR * randomDirection(2)/randomDirection(1);
% hold on;
% plot(xxR,yyR, 'b', 'LineWidth', 2);


% 4
p1 = X1*wF;
p2 = X2*wF;


plo = min([p1; p2]);
phi = max([p1; p2]);
[nn1, xx1] = hist(p1);
[nn2, xx2] = hist(p2);
hhi = max([nn1 nn2]);
figure(2),
subplot(211), bar(xx1, nn1);
axis([plo phi 0 hhi]);
title('Distribution of Projections', 'FontSize', 16)
ylabel('Class 1', 'FontSize', 14)
subplot(212), bar(xx2, nn2);
axis([plo phi 0 hhi])
ylabel('Class 2', 'FontSize', 14);

print -depsc f3-4.eps;


% 5
thmin = min([xx1 xx2]);
thmax = max([xx1 xx2]);

rocResolution = 50;
thRange = linspace(thmin, thmax, rocResolution);
ROC = zeros(rocResolution,2);
for jThreshold = 1: rocResolution
    threshold = thRange(jThreshold);
    tPos = length(find(p1 > threshold))*100 / N;
    fPos = length(find(p2 > threshold))*100 / N;
    ROC(jThreshold,:) = [fPos tPos];
end
figure(3), clf,
plot(ROC(:,1), ROC(:,2), 'b', 'LineWidth', 2);
axis([0 100 0 100]);
grid on, hold on
plot(0:100, 0:100, 'b-');
xlabel('False Positive', 'FontSize', 16)
ylabel('True Positive', 'FontSize', 16);
title('Receiver Operating Characteristic Curve', 'FontSize', 20);

print -depsc f3-5.eps;


% 6
% Compute the area under the ROC curve
areaUnderCurve = trapz(ROC(:, 2));


% 7
% For a suitable choice of decision threshold, compute the classification
% accuracy
% true negative = 100 - false positive
% If we pick a threshhold from the range and look at its TP and FP we can
% get the true negative by subtracting the FP from 100
% Our TP and TN are the ones we predicted right, so TP + TN / N will give
% us the accuracy
% When TP is 90 and FP is 58 we get:
truePositive = 90;
trueNegative = 100 - 58;
accuracyFisher = (truePositive + trueNegative) * 100 / N;
% Expand on this to a for loop to find the best threshold and also seperate
% into its own function

accuracyFisherIter = 0;
maxThreshold = 0;
for thresholdIter=1:50
    TP = ROC(thresholdIter, 2);
    TN = 100 - ROC(thresholdIter, 1);
    if ((TP + TN) * 100 / N > accuracyFisherIter)
        accuracyFisherIter = (TP + TN) * 100 / N;
        maxThreshold = thRange(thresholdIter);
    end
end

% --------------------- 8 -----------------------------

% Projecting onto a random direction
randomDirection = rand(2, 1);
[xx1F, xx2F, p1F, p2F] = projectOntoDirection(X1, X2, randomDirection);

% Computinting the ROC curve for the random direction projection
ROCF = computeROC(xx1F, xx2F, p1F, p2F, N);

% Area under ROC curve for random direction
areaUnderCurveForRandomDirection = trapz(ROCF(:, 2));

% Projecting onto the direction connecting the means of the two classes
meansDirection = m1 - m2;
[xx1M, xx2M, p1M, p2M] = projectOntoDirection(X1, X2, meansDirection);


% Compute the ROC curve for the means direction
ROCM = computeROC(xx1M, xx2M, p1M, p2M, N);

% Area under the ROC curve for the means direction
areaUnderCurveForMeansDirection = trapz(ROCM(:, 2));

% ----------------------- 9 ------------------------------------

% Implement a nearest neighbor classifier (1-NN) on this data, and compute
% its accuracy with that of the Fisher Discriminant Analyzer

% Nearest neighbour classifier
% (Caution: The following code is very inefficient)
X = [X1; X2];
N1 = size(X1, 1);
N2 = size(X2, 1);

y = [ones(N1,1); -1*ones(N2,1)];
d = zeros(N1+N2-1,1);
nCorrect = 0;

for jtst = 1:(N1+N2)
    % pick a point to test
    xtst = X(jtst,:);
    ytst = y(jtst);

    % All others form the training set
    jtr = setdiff(1:N1+N2, jtst);
    Xtr = X(jtr,:);
    ytr = y(jtr,1);

    % Compute all distances from test to training points
    for i=1:(N1+N2-1)
        d(i) = norm(Xtr(i,:)-xtst);
    end

    % Which one is the closest?
    [imin] = find(d == min(d));

    % Does the nearest point have the same class label?
    if ( ytr(imin(1)) * ytst > 0 )
        nCorrect = nCorrect + 1;
    else
        disp('Incorrect classification');
    end
end

% Percentage correst
pCorrect = nCorrect*100/(N1+N2);
disp(['Nearest neighbour accuracy: ' num2str(pCorrect)]);

% The nearest neighbor accuracy is a bit lower than the Fisher discriminant
% Analyzer. It would most likely be improved if extended to K-nearest neighbor.

% -------------------------- 10 -----------------------------------

% Construct a distance-to-mean classifier using Euclidian distance and
% Mahalanbis distance as distance measures and compare their classification
% accuracies.

% We start with the algebra for the linear classifier w'x + b gt or lt 0
w = 2 * C1^-1 * (m1 - m2);
b = (m1' * C1^-1 * m1) - (m2' * C1^-1 * m2);

% from this formula we can figure out the x and y intercepts
xIntercept = -b / w(2, 1);
yIntercept = -b / w(1, 1);

% Slope
slope = (yIntercept - 0.00) / (0.00 - xIntercept);

% Using y = mx + b for definting new x and y points on the line
xCoord1 = -3.00;
yCoord1 = slope * xCoord1 + yIntercept;

xCoord2 = 4.00;
yCoord2 = slope * xCoord2 + yIntercept;

figure(1);
hold on;
plot([xIntercept, 0], [0, yIntercept], 'g', 'LineWidth', 2);
% plot([xCoord1, xCoord2], [yCoord1, yCoord2], 'g', 'LineWidth', 2);

% Compute accuracy
%
correctMahalanobisDistance = 0;
for iter=1:400
    if ((w' * X(iter, :)' + b) * y(iter) > 0)
        correctMahalanobisDistance = correctMahalanobisDistance + 1;
    end
end
correctMahalanobisDistanceP = correctMahalanobisDistance * 100 / 400;

correctEuclidianDistance = 0;
for iter=1:400
    if (abs(X(iter, :) - m1) < abs(X(iter, :) - m2))
        predictY = 1;
    else
        predictY = -1;
    end

    if (predictY == y(iter))
        correctEuclidianDistance = correctEuclidianDistance + 1;
    end
end
correctEuclidianDistanceP = correctEuclidianDistance * 100 / 400;

% ---------------------- 11 -------------------------------------

% For the above classification problem, compute and plot a three
% dimensional graph of the posterior probabilities of one of the two
% classes for the Bayes' optimal classifier. Does the graph match your
% expectations from theory?

% Pick points in the graph and calculate their posterior probabilites.
% These points and their values values will then be used for the 3d graph.

pp1 = [4 0]';
pp2 = [4 3]';
pp3 = [1 1]';
pp4 = [1 3]';
pp5 = [-1 3]';
pp6 = [-1 1]';

pp1PP = 1 / (1 + exp(-(w'* pp1)));
pp2PP = 1 / (1 + exp(-(w'* pp2)));
pp3PP = 1 / (1 + exp(-(w'* pp3)));
pp4PP = 1 / (1 + exp(-(w'* pp4)));
pp5PP = 1 / (1 + exp(-(w'* pp5)));
pp6PP = 1 / (1 + exp(-(w'* pp6)));

% figure(1);
% hold on;
x3D = [pp1(1) pp2(1) pp3(1) pp4(1) pp5(1) pp6(1)];
y3D = [pp1(2) pp2(2) pp3(2) pp4(2) pp5(2) pp6(2)];
z3D = [pp1PP pp2PP pp3PP pp4PP pp5PP pp6PP];

ZRD = zeros(13, 13);
for i=1:13
    for j=1:13
        ZRD(i, j) = 1 / (1 + exp(-(w'* [i-7 j-7]')));
    end
end

[X3D, Y3D] = meshgrid(-6:1:6, -6:1:6);
% [X3D, Y3D] = meshgrid(x3D, y3D);
% FZ3D = 1 / (1 + exp(-(w'* [X3D Y3D])));
% FZ3D = 1 / (1 + exp(-());
gridsize = size(ZRD);
figure(4);
surf(X3D, Y3D, ZRD);

print -depsc f3-11.eps;

% ------------------------- 12 --------------------------------

% What do we expect the Bayes' optimal class boundary to be if C1 and C2
% are not identical
% Write out the algebra, change of one covariance matrices and illustrate
% the theoretical predictions.

% We know that when we calculate the Bayes' classifier for simple densities
% with distinct means and a common covariance matrix, we can simplify the
% Bayes' classifier decision rule to to the linear classifer:

% If we solve for w and b using two different covariance matrices we get
% the following:

C3 = 1.5 * eye(2);
wDistinctCovariance = (C1^-1 + C3^-1) * (m1 - m2);
bDistinctCovariance = ((m1' * C1^-1 * m2) - (m2' * C3^-1 * m2));

% Calculate the accuracy for the Bayes' optimal classifier with distint
% covariance matrices.
correctBayesDistinctCov = 0;
for iter=1:400
    if ((wDistinctCovariance' * X(iter, :)' + bDistinctCovariance) * y(iter) > 0)
        correctBayesDistinctCov = correctBayesDistinctCov + 1;
    end
end
correctBayesDistinctCovP = correctBayesDistinctCov * 100 / 400;



% ----------------- Function definitions ------------------------

function [xx1, xx2, p1, p2] = projectOntoDirection(X1, X2, direction)
    p1 = X1 * direction;
    p2 = X2 * direction;
    
    plo = min([p1; p2]);
    phi = max([p1; p2]);
    [nn1, xx1] = hist(p1);
    [nn2, xx2] = hist(p2);
    hhi = max([nn1 nn2]);
    figure(2),
    subplot(211), bar(xx1, nn1);
    axis([plo phi 0 hhi]);
    title('Distribution of Projections', 'FontSize', 16)
    ylabel('Class 1', 'FontSize', 14)
    subplot(212), bar(xx2, nn2);
    axis([plo phi 0 hhi])
    ylabel('Class 2', 'FontSize', 14);    
end

function ROC = computeROC(xx1, xx2, p1, p2, N)
   thmin = min([xx1 xx2]);
   thmax = max([xx1 xx2]);
   
   rockResolution = 50;
   thRange = linspace(thmin, thmax, rockResolution);
   ROC = zeros(rockResolution, 2);
   
   for jThreshold = 1 : rockResolution
       threshold = thRange(jThreshold);
       tPos = length(find(p1 > threshold)) * 100 / N;
       fPos = length(find(p2 > threshold)) * 100 / N;
       ROC(jThreshold, :) = [fPos, tPos];
   end
   
   figure(3);
   hold on;
   plot(ROC(:, 1), ROC(:, 2), 'r', 'LineWidth', 2);

   print -depsc f3-8.eps;
end

function aNumber = a()
    aNumber = 12;
end

