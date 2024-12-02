clc;
clear;

%Task Nr. 1: Linear Regression

load("ex1_data.mat"); %load source data
x = GestationalAge; %map to x and y
y = BirthWeight;

%VARIABLES
m = 17; %amount of data point
t0 = 0; %theta0
t1 = 0; %theta1
a = 0.001; %alpha
num_iterations = 100000;
J=1:num_iterations; %save values of the cost function for later visualization

%CALCULATION
for j =  1:num_iterations
    %calculate the sums seperately, reset to 0 every iteration
    sum0 = 0;
    sum1 = 0;
    sumCost = 0; 
    for k = 1:m %build sums
        sum0 =sum0+ (t0+t1*x(k))-y(k);
        sum1 =sum1+ ((t0+t1*x(k))-y(k))*x(k);
        sumCost = sumCost + ((t0+t1*x(k))-y(k))^2;
    end

    t0 = t0-a*(1/m)*sum0; %calculate new theta0
    t1 = t1-a*(1/m)*sum1; %calculate new theta1
    J(j) = (1/(2*m))*sumCost; %save steps of cost function in vector
end

%VISUALIZATION
tiledlayout(2,1); %split graph in 2
nexttile; %first graph
hold on; %mix 2 graphs for 1st graph (linear fn and data)
xlabel("Weeks");
ylabel("Grams");
scatter(x,y); %plot data
lin = linspace(25, 50); %plot line between 25 and 50 (Weeks)
plot(lin,t0+t1*lin);
title("Raw data");
hold off;
nexttile; %second graph
plot(1:100, J(1:100)); %truncate the visualization of the cost function as barely anything changes after 100 iterations, but you can see the convergence before 100 a lot easier
title("Cost function");
