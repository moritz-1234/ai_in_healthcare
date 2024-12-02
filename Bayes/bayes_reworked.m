clc;
clear;
T = readtable("Naive Bayes classifier task - training group.txt");
t = table2array(T);

data = t;
%removed data smoothing

c = corr(data(:,1:17)); %calculate correlation table for anything but the result
c((c>=-0.6)&(c<=0.6)) = NaN; %filter out not strongly correlated values (-0,6<c<0,6)
isupper = logical(triu(ones(size(c)),1)); %filter out mirrored (upper) part https://se.mathworks.com/matlabcentral/answers/1750730-how-to-plot-correlation-coefficient-matrix-plot
c(isupper) = NaN;
h = heatmap(c,'MissingDataColor','w');
% number identifies the column, starting indexing at 1(e.g. 2: age)

%you can see: strong correlation between 2,12; 3,4; 4,5; 9,18;
%11,13;11,14; 13,14; 16,17
%also: strong negative correlation between 1 and 3; as 3 is removed later
%this does not change anything

% The strongest correlation is between 11 and 13/14

% Conclusion:
% - 11, 13 and 14 are strongly correlated to each other => remove 13, 14
% - 16 and 17 are strongly correlated => remove 16
% - 3,4 as well as 4,5 are strongly correlated => remove 3, 5
% - 2, 12 are strongly correlated => remove 12
clf; %remove previous figure(set breakpoint here to see full data correlation)

data(:, 13) = []; % remove 13
data(:, 13) = []; % remove 14 (as 13 is removed now it is the new 13)
data(:, 14) = []; % remove 16
data(:, 3) = []; %remove 3
data(:, 4) = []; %remove 5
data(:, 10) = []; %remove 12

c2 = corr(data(:,1:11)); %double check new data
c2((c2>=-0.6)&(c2<=0.6)) = NaN; %filter out not strongly correlated values (-0,6<c2<0,6)
isupper = logical(triu(ones(size(c2)),1)); %filter out mirrored (upper) part https://se.mathworks.com/matlabcentral/answers/1750730-how-to-plot-correlation-coefficient-matrix-plot
c2(isupper) = NaN;
h = heatmap(c2,'MissingDataColor','w'); %display new heatmap to double check data is okay now
clf('reset'); %set breakpoint here to check corrected data correlation
close all; %close figure window (here not needed anymore)
%result: no strong positive / negative correlations anymore

%discretize continuous values

%new constants
FEATURE_COUNT = 12;
m = 46;
N = 8; %amount of bins the data is discretized into

%discretize continuos values
p_x_c = 1:FEATURE_COUNT; %save percentages for each feature; prepare vector
for i = 1:FEATURE_COUNT %discretize all continuous features
    % continuous features: 2-4, 7-11
    if((i >=2 && i <=4) || (i >=7 && i<=11))
        edges = linspace(min(data(:,i)), max(data(:,i)),N); %equal sized bins between min, max of original data
        values = edges(2:end); %asign values instead of categories
        bins = discretize(data(:,i),edges,values); %group data into equal sized bins
        data(:, i) = bins;
    end
end

%build final bayes probabilities
dataable = array2table(data);
diseaseData = table2array(groupsummary(dataable, 12));
totalDisease = diseaseData(2,2); %total number of sick patients (16)
totalHealthy = diseaseData(1,2); %total number of healthy patients(30)

p_x = NaN(8,3*(FEATURE_COUNT-1));%result table;(first feature then p_x_c1, then p_x_c2)*features NAN as default for debugging and to differentiate 0 values from empty ones
for i = 1:FEATURE_COUNT-1 %-1 as do not calculate for result column
    G = table2array(groupsummary(dataable,[i, 12], 'IncludeEmptyGroups', true));%first column: disease, second i, third total
    categories = size(G,1)/2; % size(g) is double the amount of categories as there is a value for each outcome (healthy, sick)
    for j = 1:2:size(G,1) %for every category
        % j uneven: healthy, j even: sick
        index = floor((j+1)/2); %store in the right index in the result array
        p_x(index,3*i-2) = G(j,1); % save feature
        p_x(index,3*i-1)= ((G(j,3)+1) / (totalHealthy+categories)); %save probability and apply laplace smoothing (+1). as you do this for every category, later add the amount of categories in the division 
        p_x(index,3*i) = ((G(j+1,3)+1) / (totalDisease+categories)); %save probability and apply laplace smoothing (+1)
    end
end
smoothedP = smoothdata(p_x(:,2));

p_c_1 = (totalHealthy)/(m);
p_c_2 = (totalDisease)/(m);

%use test data

test_data_raw = readtable("Naive Bayes classifier task - test group.txt");
test_data = table2array(test_data_raw);

%remove the same columns as in the training data
test_data(:, 13) = []; % remove 13
test_data(:, 13) = []; % remove 14 (as 13 is removed now it is the new 13)
test_data(:, 14) = []; % remove 16
test_data(:, 3) = []; %remove 3
test_data(:, 4) = []; %remove 5
test_data(:, 10) = []; %remove 12


for i = 1:size(test_data,1) %for each patient
    multc1 = 0;
    multc2 = 0;
    for j= 1:FEATURE_COUNT-1% for every feature (other then the result)
        comparison = test_data(i,j);
        [~, index] = min(abs(p_x(:,j*3-2) - test_data(i, j))); %find the closest index of the raw value in the test data to the discretized value
        p_c1_f = p_x(index,j*3-1); %probability of being healthy for that feature
        p_c2_f = p_x(index, j*3); %probability of being sick for that feature
        multc1 = multc1 + log(p_c1_f); % add logarithms together to avoid underflow
        multc2 = multc2+log(p_c2_f); % add logarithms together to avoid underflow
    end
    %corrected logarithm calculation from product to sum
    res_p_c1 = log(p_c_1) + multc1; %result probabilty of being healthy
    res_p_c2 = log(p_c_2) + multc2; % result probability of being sick
    outcome = "";
    if res_p_c1 > res_p_c2 %healthy is bigger than sick
        outcome = "healthy";
    elseif res_p_c2 >res_p_c1 %sick is bigger than healthy
        outcome ="sick";
    else %both are equal
        outcome ="cannot tell";
    end
    fprintf("Patient number %d has chance of healthyness: %f and chance of sickness: %f so the result is: %s\n", i, res_p_c1, res_p_c2, outcome);
end
%result: all 3 patients are sick










