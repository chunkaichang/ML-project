%--------------------------------------------------------------%
% Machine Learning Spring2014 Final Project - Galaxy zoo       %
%--------------------------------------------------------------%
% Update: 2014/06/22 (Scale, parameter finding loop)
% Update: 2014/06/23 (Mean Square Error compute automatically)

% Reset and loading data
clear all; clc;
load('X_train.mat'); load('X_test.mat'); % load('X_test2');
X_train = X_train';
X_test = X_test';
% X_train = X_train(1:2000,:);
% X_test = X_test(1:330,:);
% X_test = X_test2';

% data scale to [0 1]
X_train = (X_train - repmat(min(X_train,[],1),size(X_train,1),1))...
   *spdiags(1./(max(X_train,[],1)-min(X_train,[],1))',0,size(X_train,2),size(X_train,2));
X_test = (X_test - repmat(min(X_test,[],1),size(X_test,1),1))...
   *spdiags(1./(max(X_test,[],1)-min(X_test,[],1))',0,size(X_test,2),size(X_test,2));

% X_test2_sc = (X_test2 - repmat(min(X_test2,[],1),size(X_test2,1),1))...
%    *spdiags(1./(max(X_test2,[],1)-min(X_test2,[],1))',0,size(X_test2,2),size(X_test2,2));

% parameter declaration
N = size(X_train,1);
N2= size(X_test,1);
offset = 15000;
% offset2= 43690;

Data_Target = csvread('training_solutions_rev1.csv',1,0);
T_train = Data_Target(1:N,2:4);
T_test = Data_Target(offset+1:offset+N2,2:4);
% T_test2= Data_Target(offset2+1:offset2+N2,2:4);

% This is a parameter finding loop 

% bestcv = 0;
% for log2c = -1:6,
%   for log2g = -6:1,
%     fprintf('now we are testing:c=%g, g=%g \n', 2^log2c, 2^log2g);
%     cmd = ['-s 3 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
%     cv = svmtrain(T_train(:,1), X_train, cmd);
%     cv2= svmtrain(T_train(:,2), X_train, cmd);
%     if (cv+cv2 >= bestcv),
%       bestcv = cv+cv2; bestc = 2^log2c; bestg = 2^log2g;
%     end
%     fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv+cv2, bestc, bestg, bestcv);
%   end
% end

% start training & testing
options = ['-s 3 -t 1 -b 1 -c 1 -g 0.0001'];
tic
model_1 = svmtrain(T_train(:,1), X_train, options);
model_2 = svmtrain(T_train(:,2), X_train, options);
toc
[predict_label_1, accuracy_1, dec_values_1] = svmpredict(T_test(:,1), X_test, model_1);
[predict_label_2, accuracy_2, dec_values_2] = svmpredict(T_test(:,2), X_test, model_2);
predict_label_3 = ones(N2,1);
predict_label_3 = predict_label_3 - predict_label_1 - predict_label_2;

% compute MSE
mse3 = 0;
for ii=1:N2
    mse3=mse3 + (predict_label_3(ii,1)- T_test(ii,3))^2;
end
mse3 = mse3/N2;
Total_MSE = mse3 + accuracy_1(2,1) + accuracy_2(2,1);
display(Total_MSE);


%--------------------------------------------------------------------
%--------------------------------------------------------------------
%--------------------------------------------------------------------
