% load financial data csv file
FinanceData = csvread('table-5.dat', 1, 0);

% Grap the closing index value
closing_index = FinanceData(:, 5);
N_closing_index = size(closing_index, 1);

% Predict tomorrow's index value from past 20 trading days.

% Construct our design matrix
design_matrix = ones(N_closing_index - p, p);

for i=1:size(design_matrix, 1)
    n = (p + 1) + (i - 1);
    for j=1:p 
      design_matrix(i, j) = closing_index(n - j, 1);
    end
end

% Formulate a neural network predictor that predicts tomorrows index value
actual_values_to_design_matrix = closing_index(p + 1:N_closing_index, 1);

net = feedforwardnet(20);
net = train(net, design_matrix', actual_values_to_design_matrix');
view(net);
predicted_values = net(design_matrix');

figure(14)
plot([1:size(design_matrix, 1)]', predicted_values,[1:size(design_matrix, 1)]', actual_values_to_design_matrix);

