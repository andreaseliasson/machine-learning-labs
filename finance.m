% load financial data csv file
FinanceData = csvread('table-5.dat', 1, 0);

% Grap the closing index value
closing_index = FinanceData(:, 5);
N_closing_index = size(closing_index, 1);

% Predict tomorrow's index value from past 20 trading days.
p = 20;

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

figure(20)
plot([1:size(design_matrix, 1)]', predicted_values,[1:size(design_matrix, 1)]', actual_values_to_design_matrix),
title('Predicted and Actual Index Values', 'FontSize', 14);
print -depsc fa-11.eps;

% Split into training and test sets

training_set = closing_index(1:900, 1);
training_set_outputs = closing_index(1:900, 1);

test_set = closing_index(900 + 1:size(closing_index, 1), 1);
test_set_outputs = closing_index(900 + 1: size(closing_index, 1), 1);

training_set_design_matrix = ones(size(training_set, 1) - p, p);
training_set_design_matrix_outputs = closing_index(p + 1:900, 1);

for i=1:size(training_set_design_matrix, 1)
    n = (p + 1) + (i - 1);
    for j=1:p 
      training_set_design_matrix(i, j) = closing_index(n - j, 1);
    end
end

% Construct test set design matrix
test_set_design_matrix = ones(size(test_set, 1) - p, p);
test_set_design_matrix_outputs = closing_index(900 + p + 1:size(closing_index), 1);

for i=1:size(test_set_design_matrix, 1)
   n = (p + 1) + (i - 1);
   for j=1:p 
      test_set_design_matrix(i, j) = closing_index((900 + n) - j, 1);
   end
end

net = feedforwardnet(20);
net = train(net, training_set_design_matrix', training_set_design_matrix_outputs');
view(net);
test_set_predicted_values = net(test_set_design_matrix')';

figure(21),
plot([1:size(test_set_predicted_values, 1)]', test_set_design_matrix_outputs, [1:size(test_set_predicted_values, 1)]', test_set_predicted_values);
title('Nueral Net on unseen data (test set)', 'FontSize', 14);
print -depsc fa-12.eps;

% Long term iterated prediction

number_of_iterations = 20;

long_term_design_matrix = ones(number_of_iterations, p);

predicted_value_of_row = ones(number_of_iterations, 1);

test_set_n_outputs = test_set_outputs(1:number_of_iterations, 1);

for i=1:number_of_iterations
    if (i == 1)
        for j=1:p
            long_term_design_matrix(i, j) = training_set_design_matrix_outputs(size(training_set_design_matrix, 1)+1-j);
        end
        predicted_value_of_row(i) = net(long_term_design_matrix(i, :)');
    end
    if (i > 1)
        for j=1:p
            if (j == 1)
                long_term_design_matrix(i, j) = predicted_value_of_row(i-1);
            else
                long_term_design_matrix(i, j) = long_term_design_matrix(i-1,j-1);
            end
        end
        predicted_value_of_row(i) = net(long_term_design_matrix(i, :)');
    end
end

figure(22),
plot([1:number_of_iterations]', test_set_n_outputs, [1:number_of_iterations]', predicted_value_of_row),
legend('actual index values','predicted index values'),
title('Long term iterated prediction', 'FontSize', 14);

% Include past values of volume traded


