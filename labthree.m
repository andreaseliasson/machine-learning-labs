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


%2
% Draw 200 samples from each of the two distributions and plot them on the
% top of the countours


N = 200;
X1 = mvnrnd(m1, C1, N);
X2 = mvnrnd(m2, C2, N);
plot(X1(:, 1), X1(:, 2), 'bx', X2(:, 1), X2(:, 2), 'ro');
grid on;


% 3
% Compute the Fisher Linear Discriminant direction using the means and
% covariance matrcices of the problem, and plot the discriminant direction.


wF = inv(C1+C2)*(m1-m2);
xx = -6:0.1:6;
yy = xx * wF(2)/wF(1);
plot(xx,yy, 'r', 'LineWidth', 2);


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


% 6
% Compute the area under the ROC curve
areaUnderCurve = trapz(ROC(:, 2));


% 7
% For a suitable choice of decision threshold, compute the classification
% accuracy
% true negative = 100 - false positive
trueNegative = 100 - 67.5;
errorRate = trueNegative / 200 * 100;
% Expand on this to a for loop to find the best threshold and also seperate
% into its own function

% 8
