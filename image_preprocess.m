%--------------------------------------------------------------%
% Machine Learning Spring2014 Final Project - Galaxy zoo       %
%--------------------------------------------------------------%
% image_preprocess.m
% Reset
clear all; clc;

% parameter declaration
N = 1000;
sample_rate = 4;
D = (200/sample_rate)*(200/sample_rate);
x = zeros(D,N);
%% Training set
% data feature: x (D by N) a data structure that contains one training example per column
% D: dimension of image; N: training data size

for i = 1:N
    % file_name = strcat('image',num2str(i),'.jpg');
    file_name = strcat('image (',num2str(i),').jpg');
    ImageRead = imread(file_name);
    cut       = ImageRead(112:311,112:311,:);% crop the middle 200x200 part
    gray_pic  = rgb2gray(cut);
    down = gray_pic(1:sample_rate:end,1:sample_rate:end); % downsample
    x(:,i) = double(reshape(down,[],1));%horizontally cat input
end
%% Test set
N2 = 200;
offset = 50000;
y = zeros(D,N2);
for i = 1:N2
    % file_name = strcat('image',num2str(i),'.jpg');
    file_name = strcat('image (',num2str(i+offset),').jpg');
    ImageRead = imread(file_name);
    cut       = ImageRead(112:311,112:311,:);
    gray_pic  = rgb2gray(cut);
    down = gray_pic(1:sample_rate:end,1:sample_rate:end); 
    y(:,i) = double(reshape(down,[],1));
end
%% mean normalization (-zero mean)
avg_train = mean(x,2);
x = x - repmat(avg_train, 1, size(x, 2));
avg_test = mean(y,2);
y = y - repmat(avg_test, 1, size(y, 2));
%% compute covariance matrix of training set
if(D > N) % high-dimension data
    sigma = x' * x / size(x, 2);
    % compute eigenvectors (columns of U: in decreasing order of eigenvalues
    [U,S,V] = svd(sigma);
    % compute the dimension needed for retaining 99% variance
    ev = diag(S); % extract eigenvalues from S
    for i = 1: size(ev,1)
        varpercent = sum(ev(1:i))/sum(ev);
        if(varpercent > 0.98)
            break; 
        end
    end
    M = i;
    lambda = diag(1./sqrt(N*ev(1:M)));
    U = x * U(:,1:M)*lambda; % normalized eigenvector
    ev = ev(1:M);
end
if(D <= N)
    sigma = x * x' / size(x, 2);
    [U,S,V] = svd(sigma);
    ev = diag(S);
    for i = 1: size(ev,1)
        varpercent = sum(ev(1:i))/sum(ev);
        if(varpercent > 0.98)
            break; 
        end
    end
    M = i;
    U = U(:,1:M);
    ev = ev(1:M);
end    
%% Whitening
epsilon = 1e-05; % avoid dividing by zero
X_train = diag(1./sqrt(ev + epsilon)) * U' * x;
X_test = diag(1./sqrt(ev + epsilon)) * U' * y;
save('X_train.mat','X_train');
save('X_test.mat','X_test');

%% Recover image
% use the first image as an example
% r_train = U * (x(:,1)' * U)' + avg_train;
% r_test = U * (y(:,1)' * U)' + avg_test;
% imwrite(uint8(reshape(r_train,50,50)),'recover.jpg');
% imwrite(uint8(reshape(r_test,50,50)),'recover2.jpg');
