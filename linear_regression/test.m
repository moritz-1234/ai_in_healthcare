% Generate synthetic data (you'll need your own dataset)
importdata("ex1_data.mat");

gestational_age = GestationalAge; % Weeks
birth_weight = BirthWeight; % Grams

% Initialize model parameters
theta0 = 0; % Intercept
theta1 = 0; % Slope

% Hyperparameters
alpha = 0.01; % Learning rate
num_iterations = 1000;

% Gradient descent
for iter = 1:num_iterations
    % Compute predictions
    predictions = theta0 + theta1 * gestational_age;
    
    % Compute errors
    errors = predictions - birth_weight;
    
    % Update parameters
    temp0 = theta0 - alpha * sum(errors) / length(gestational_age);
    temp1 = theta1 - alpha * sum(errors .* gestational_age) / length(gestational_age);
    
    % Update simultaneously
    theta0 = temp0;
    theta1 = temp1;
    
    % Compute cost (mean squared error)
    cost = sum(errors.^2) / (2 * length(gestational_age));
    
    % Display cost at each step
    fprintf('Iteration %d, Cost: %.4f\n', iter, cost);
end

% Plot data and linear fit
scatter(gestational_age, birth_weight, 'b', 'filled');
hold on;
x_vals = min(gestational_age):max(gestational_age);
y_vals = theta0 + theta1 * x_vals;
plot(x_vals, y_vals, 'r', 'LineWidth', 2);
xlabel('Gestational Age (weeks)');
ylabel('Birth Weight (grams)');
title('Linear Regression: Birth Weight vs. Gestational Age');
legend('Data', 'Linear Fit');
grid on;
hold off;
