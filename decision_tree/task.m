clc;
clear;
clf;

load('AccSignal.mat');

figure(1);
hold on;
plot(accX);
plot(accY);
plot(accZ);
for n = 1:size(accSegX,2)
    featuresTrain(n,1) = mean(abs(accSegX(:, n)));
    featuresTrain(n,2) = mean(abs(accSegY(:, n)));
    featuresTrain(n,3) = mean(abs(accSegZ(:, n)));
    featuresTrain(n, 4) = std(accSegX(:, n));
    featuresTrain(n, 5) = std(accSegY(:, n));
    featuresTrain(n, 6) = std(accSegZ(:, n));
end

classifierTree = fitctree(featuresTrain, accLabel);
view(classifierTree,'mode','graph');

% load test data
load("AccSignalVal.mat");
%other solutions: divide training data e.g. 70/30
for n = 1:size(accSegXVal,2)
    featuresTest(n,1) = mean(abs(accSegXVal(:, n)));
    featuresTest(n,2) = mean(abs(accSegYVal(:, n)));
    featuresTest(n,3) = mean(abs(accSegZVal(:, n)));
    featuresTest(n, 4) = std(accSegXVal(:, n));
    featuresTest(n, 5) = std(accSegYVal(:, n));
    featuresTest(n, 6) = std(accSegZVal(:, n));
end
labels = predict(classifierTree,featuresTest); %classified data
cf = confusionmat(accLabelVal,labels);
hold off;
confusionchart(cf);

%Results: 
% 1: 36  TP, 84 FN, 7 FP, TN = 660-(36+84+7) = 533
% 2: 250 TP, 2 FN, 38 FP, TN = 660-(250+2+38)= 370
% 3: 178 TP, 2 FN, 48 FP, TN = 660-(178+2+48)= 432
% 4: 103 TP, 5 FN, 0 FP , TN = 660-(103+5+0) = 552

tp = 1:4;
fp = 1:4;
tn = 1:4;
fn = 1:4;
FP = 0;
for i =1:4 %TP, FN
    FN = 0;
    for j = 1:4        
        if(i == j) %TP
            tp(i) = cf(i, j);
        else %FN
            FN = FN + cf(i, j);
        end
    end
    fn(i) = FN;
end
for i =1:4 %FP 
    FP = 0;
    for j = 1:4        
        if(i ~=j)
            FP = FP + cf(j, i);
        end
    end
    fp(i) = FP;
end
for i = 1:4 %TN
    tn(i) = 660-fn(i)-tp(i)-fp(i);
end


for i = 1:4 %results
    TPR = tp(i)/(tp(i)+fn(i));
    TNR = tn(i) / (tn(i)+fp(i));
    FPR = fp(i) / (fp(i)+tn(i));
    FNR = fn(i) / (fn(i)+tp(i));
    ACC = (tp(i)+tn(i))/(tp(i)+tn(i)+fp(i)+fn(i));
    fprintf("For parameter %d, the model has TPR = %f, TNR = %f, FPR = %f, FNR = %f and accuracy = %f\n",i,TPR, TNR, FPR, FNR, ACC);
end

%Results:
% For type 1, the model has TPR = 0.300000, TNR = 0.987037, FPR = 0.012963, FNR = 0.700000 and accuracy = 0.862121
% For type 2, the model has TPR = 0.992063, TNR = 0.906863, FPR = 0.093137, FNR = 0.007937 and accuracy = 0.939394
% For type 3, the model has TPR = 0.988889, TNR = 0.900000, FPR = 0.100000, FNR = 0.011111 and accuracy = 0.924242
% For type 4, the model has TPR = 0.953704, TNR = 1.000000, FPR = 0.000000, FNR = 0.046296 and accuracy = 0.992424

% Conclusion: It performed the best for activity 4, the worst for activity 4. In general, the accuracy was pretty high though.  






