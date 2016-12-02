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
plot(m2(1), m2(2), 'r*', 'LineWidth', 4);

% Show 100 samples drawn from each of the classes superposed on the same
% graph
N = 100;
X1 = mvnrnd(m1, C1, N);
X2 = mvnrnd(m2, C2, N);

hold on;
plot(X1(:, 1), X1(:, 2), 'bx', X2(:, 1), X2(:, 2), 'ro');
grid on;

% Draw the posterior probability as a three dimensional graph
figure(2)
mesh(xRange, yRange, P1./(P1+P2));

% Using the data sampled from each of the distributions, train a
% feedforward nural network using the Neural Networks toolbox. 

XNN = [X1; X2]';
N1 = size(X1, 1);
N2 = size(X2, 1);
y1 = [ones(N1, 1) zeros(N2, 1)]';
y2 = [zeros(N2, 1) ones(N1, 1)]';
y = [y1 y2];

net = patternnet(20);
net = train(net, XNN, y);
view(net);
output = net(XNN);

NP1 = zeros(numGrid, numGrid);

% ---------- can be uncommented -----------

% for i=1:numGrid
%     for j=1:numGrid
%         x = [yRange(j) xRange(i)]';
%         nnp = net(x);
%         NP1(i,j) = nnp(1);
%     end
% end
% 
% NPmax = max(max(NP1));
% figure(3), clf, contour(xRange, yRange, NP1, [0 0.5*NPmax], 'LineWidth', 2);
% hold on;
% plot(m1(1), m1(2), 'b*', 'LineWidth', 4);
% plot(m2(1), m2(2), 'r*', 'LineWidth', 4);
% 
% hold on;
% plot(X1(:, 1), X1(:, 2), 'bx', X2(:, 1), X2(:, 2), 'ro');
% grid on;

% ---------------------- Can be uncommented ------------------------


% ------------------------ Time series prediction --------------------
% Time series prediction
% Formulated as a regression problem.

Ttr = T(1:1500, 1);
Tts = T(1500 + 1: size(T, 1), 1);
NTtr = size(Ttr, 1);
NTts = size(Tts, 1);
p = 20;

ytr1 = X(1:1500, 1);
ytr = X(p+1:NTtr, 1);
yts = X(1500+p+1:2001, 1);

D = ones(NTtr - p, p);

for i=1:NTtr - p
   n = (p + 1) + (i - 1);
   for j=1:p
      % We might want to change this to use the actual ouptuts from
      % the previous time series. 
      D(i, j) = X(n - j, 1);
   end
end

% D = [ones(NTtr - p, 1) D];

w = D \ ytr;
fts = D * w;

% figure
% plot(Ttr, ytr1);
% % set(gca,'xlim',[0, Ttr(end)]);
% xlabel('t');
% ylabel('x(t)');
% title(sprintf('A Mackey-Glass time serie (tau=%d)', tau));

figure(10);
plot(Ttr(p+1:NTtr, 1), fts, Ttr(p+1:NTtr, 1), ytr);
figure;
% plot(Ttr(p+1:NTtr, 1), ytr);

% Test data
D2 = ones(NTts - p, p);

for i=1:NTts - p
   n = (p + 1) + (i - 1);
   for j=1:p 
      D2(i, j) = X((1500 + n) - j, 1);
   end
end

fts2 = D2 * w;

net2 = feedforwardnet(20);
net2 = train(net2, D', ytr');
view(net2);
output2 = net2(D');

figure(11);
plot(Ttr(p+1:NTtr, 1), output2, Ttr(p+1:NTtr, 1), ytr);

% Iterate through the number of test data, and append a new row using the
% predicted value from the previous row as the first input into the new
% row. Repeat for number of test data times. Plot the predicted values
% against the test set output values.

long_term_prediction_design_matrix = ones(NTts, p);
long_term_prediction = ones(NTts, 1);
predicted_value_of_row = ones(NTts, 1);
% predictied_value_of_previous_row = net2(D(size(D, 1), :)');
for i=1:NTts
    if (i == 1)
        for j=1:p
            long_term_prediction_design_matrix(i, j) = fts(size(fts, 1)+1-j);
        end
        predicted_value_of_row(i) = net2(long_term_prediction_design_matrix(i, :)');
    end
    if (i > 1)
        for j=1:p
            if (j == 1)
                long_term_prediction_design_matrix(i, j) = predicted_value_of_row(i-1);
            else
                long_term_prediction_design_matrix(i, j) = long_term_prediction_design_matrix(i-1,j-1);
            end
        end
        predicted_value_of_row(i) = net2(long_term_prediction_design_matrix(i, :)');
    end
end

figure(12);
plot(T(p+1:size(T, 1), 1), [fts; predicted_value_of_row]);

total_predictions = [fts; predicted_value_of_row];
pr_errors = ones(size(T, 1) - p, 1);

yts_all = X(1500:size(X, 1), 1);
f_output = [ytr; yts_all];
for i=1:size(total_predictions, 1)
    pr_errors(i) = total_predictions(i) - f_output(i);
end


% Plot prediction errors
figure(13)
plot(T(p+1:size(T, 1), 1), pr_errors);