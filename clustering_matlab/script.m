
clear;
close all;
clc;
clf;
load('data.mat');

age = data(:,1);
bmi = data(:,2);
physicalHealth = data(:,3);
psychologicalHealth = data(:,4);
iod = data(:,5);


data(:, 1) = []; %remove age
data(:, 1) = []; %remove bmi


Smax = 0;
kMax = 0;
for i= 2:30 %arbitrary choice: try every clustering until 30
    [idx, C] = kmeans(data, i);
    S = silhouette(data,idx);
    s = mean(S);
    if(s>Smax)
                kMax = i;
        Smax = s;

    end
end

disp(kMax);
disp(Smax);

%result: maximum silhuette value for 2 clusters
%with all values: maximum silhuette value 0.4851
%now try to increase it by removing one column
%trial and error: maximum reached when removing age and bmi:  SMax = 0.6173






