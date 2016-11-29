% Compute the posterior probability on a regular grid in the input space and plot the decision
% boundary for which the posterior probability satisfies P[?1 | x] = 0.5. Show 100 samples
% drawn from each of the classes superposed on the same graph. Draw the posterior probability
% as a three dimensional graph.

m1 = [0;3];
C1 = [2 1; 1 2];

m2 = [2;1];
C2 = [1 0; 0 1];

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
figure(1), clf, contour(xRange, yRange, P1./(P1+P2), [0 0.5*Pmax], 'LineWidth', 2);
hold on;
plot(m1(1), m1(2), 'b*', 'LineWidth', 4);
contour(xRange, yRange, P2, [0.1*Pmax 0.5*Pmax 0.8*Pmax], 'LineWidth', 2);
plot(m2(1), m2(2), 'r*', 'LineWidth', 4);




% N = 100;
% X1 = mvnrnd(m1, C1, N);
% X2 = mvnrnd(m2, C2, N);
% 
% figure(1);
% hold on;
% plot(X1(:, 1), X1(:, 2), 'bx', X2(:, 1), X2(:, 2), 'ro');
% grid on;
% 
% W1 = -0.5 * inv(C1);
% w1 = inv(C1) * m1;
% w10 = -0.5 * m1' * inv(C1) * m1 - 0.5 * log(det(C1));
