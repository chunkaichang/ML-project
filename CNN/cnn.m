% cnn.m - CNN with two conv-subsampling layers and two fully-connected
% layers
% whiten image version
% CL: convolution layer;
% SSL: subsampling layer(max pooling)
clear all; clc;
load('X_train_white.mat');
load('T_train.mat');
D = 50;% dimension of input image
N = 1000; % number of images
train_ratio = 0.7;
num_train = N * train_ratio;
num_val = N - num_train; % validation set (for early stopping)
min_epoch = 0;
epoch = 1000;
learn_rate = 0.01;
fix_learn_rate = 0.01;
min_val_error = Inf;
GL_tolerance = 2; %generalization loss ratio
%% Convolution layer matrix init
% CL1
k1 = 15; % kernel width of CL1
num_fm1 =4; % number of feature maps of CL1
[W_CL1,B_CL1,A_CL1,Z_CL1,D_CL1,dim_CL1,opt_W_CL1,opt_B_CL1] = CL_init(k1, num_fm1, 1, D);
% SSL1
dim_SSL1 = dim_CL1/2;
A_SSL1 = zeros(dim_SSL1,dim_SSL1,num_fm1);
D_SSL1 = zeros(dim_SSL1,dim_SSL1,num_fm1);
I_SSL1 = zeros(dim_SSL1,dim_SSL1,num_fm1); % index for max values in CL
% CL2
k2 = 5;
num_fm2 =6;
[W_CL2,B_CL2,A_CL2,Z_CL2,D_CL2,dim_CL2,opt_W_CL2,opt_B_CL2] = CL_init(k2, num_fm2, num_fm1, dim_SSL1);

% SSL2
dim_SSL2 = dim_CL2/2;
A_SSL2 = zeros(dim_SSL2,dim_SSL2,num_fm2);
D_SSL2 = zeros(dim_SSL2,dim_SSL2,num_fm2);
I_SSL2 = zeros(dim_SSL2,dim_SSL2,num_fm2);
%% fully-connected layers init
num_in = numel(A_SSL2); 
num_out = 3;
% M : number of hidden unit (excluding bias unit)
% N : training data size(number of images)
M1 = 100;     M2 = 80;
batch_size = 1;
num_batch = floor(N/batch_size);

error = zeros(epoch,1);% training error
avg_train_err = zeros(epoch,1);
avg_val_err = zeros(epoch,1);
% learn_rate = 1/sqrt(batch_size)/20;
momentum = 0;
% 
% % weight matrices 
w_i = fan_in_out_rand(num_in,M1,[M1,num_in]);       b_i = zeros(M1, 1);
w_h = fan_in_out_rand(M1,M2,[M2, M1]);              b_h = zeros(M2, 1);
w_h2 = fan_in_out_rand(M2,num_out,[num_out, M2]);   b_h2 = zeros(num_out, 1);
opt_w_i = zeros(M1,num_in);         opt_b_i = zeros(M1, 1);
opt_w_h = zeros(M2, M1);            opt_b_h = zeros(M2, 1);
opt_w_h2 = zeros(num_out, M2);      opt_b_h2 = zeros(num_out, 1);
% w_i = normrnd(0,1,M1, num_in);     b_i = normrnd(0,1,M1, 1);
% w_h = normrnd(0,1,M2, M1);         b_h = normrnd(0,1,M2, 1);
% w_h2 = normrnd(0,1,num_out, M2);   b_h2 = normrnd(0,1,num_out, 1);

%% Training
for e = 1 : epoch
    % Here, batch_size = 1. Therefore, num_batch = N.
    for bat = 1 : num_train
        % load whiten img
        img = reshape(X_train(:,bat),D,D);
        % Convolutional layers forward
        % CL1 and SSL 1 
        for f = 1 : num_fm1
            A_CL1(:,:,f) = conv2(img, rot90(W_CL1(:,:,f),2),'valid') + repmat(B_CL1(f),dim_CL1,dim_CL1);
            Z_CL1(:,:,f) = sigmoid(A_CL1(:,:,f));
            [A_SSL1(:,:,f), I_SSL1(:,:,f)]= max_pool(Z_CL1(:,:,f));
        end
        % CL2 and SSL 2 
        for f = 1 : num_fm2
%             for ff = 1 : num_fm1
%                 A_CL2(:,:,f) = A_CL2(:,:,f) + conv2(A_SSL1(:,:,ff), rot90(W_CL2(:,:,f),2),'valid') + repmat(B_CL2(f),dim_CL2,dim_CL2);
%             end
            A_CL2(:,:,f) = conv2(sum(A_SSL1,3), rot90(W_CL2(:,:,f),2),'valid') + repmat(B_CL2(f),dim_CL2,dim_CL2);
            Z_CL2(:,:,f) = sigmoid(A_CL2(:,:,f));
            [A_SSL2(:,:,f), I_SSL2(:,:,f)]= max_pool(Z_CL2(:,:,f));
        end    
        % Fully-connected forward
        in = reshape(A_SSL2,1,num_in);
        sT_train = T_train(bat,:);
        % Forword to hidden layer 1 
        a_h = w_i * in' + repmat(b_i, 1, batch_size);
        z_h = sigmoid(a_h);
        % Forword to hidden layer 2
        a_h2 = w_h * z_h + repmat(b_h, 1, batch_size);
        z_h2 = sigmoid(a_h2);
        % Forword to output layer
        a_out = w_h2 * z_h2 + repmat(b_h2, 1, batch_size);
        delta_o = a_out(1:num_out,:) - sT_train';
        % Compute error
        error(e,1) = error(e,1) + 0.5*sum((a_out' - sT_train).^2);
        % Backpropagate to hidden layer 2
        dz_da2 = z_h2.*(1.0 - z_h2);
        delta_h2 = dz_da2.*(w_h2' * delta_o);
        % Backpropagate to hidden layer 1
        dz_da = z_h.*(1.0 - z_h);
        delta_h = dz_da.*(w_h' * delta_h2);
        % Backprop to SSL2
        D_SSL2 = reshape((w_i' * delta_h),dim_SSL2,dim_SSL2,num_fm2);
        % Backprop to CL2
        temp_CL2 = zeros(k2,k2,num_fm2);
        for f = 1 : num_fm2
            D_CL2(:,:,f) = SSL_backprop(D_CL2(:,:,f),D_SSL2(:,:,f),I_SSL2(:,:,f));
            B_CL2(f) = B_CL2(f) - learn_rate*sum(sum(D_CL2(:,:,f)));
%             for ff = 1 : num_fm1
%                 temp_CL2(:,:,f) = temp_CL2(:,:,f) + rot90(conv2(A_SSL1(:,:,ff),rot90(D_CL2(:,:,f),2),'valid'),2);
%             end
%             W_CL2(:,:,f) = W_CL2(:,:,f) - learn_rate .*temp_CL2(:,:,f);
            W_CL2(:,:,f) = W_CL2(:,:,f) - learn_rate .* rot90(conv2(sum(A_SSL1,3),rot90(D_CL2(:,:,f),2),'valid'),2);
        end
        % Backprop to SSL1
        for f = 1 : num_fm1
%             for ff = 1 : num_fm2
%                 D_SSL1(:,:,f) = D_SSL1(:,:,f) + conv2(D_CL2(:,:,ff),rot90(W_CL2(:,:,ff),2),'full');
%             end
            D_SSL1(:,:,f) = D_SSL1(:,:,f) + conv2(sum(D_CL2,3),rot90(sum(W_CL2,3),2),'full');
        end
        % Backprop to CL1
        for f = 1 : num_fm1
            D_CL1(:,:,f) = SSL_backprop(D_CL1(:,:,f),D_SSL1(:,:,f),I_SSL1(:,:,f));
            B_CL1(f) = B_CL1(f) - learn_rate*sum(sum(D_CL1(:,:,f)));
            W_CL1(:,:,f) = W_CL1(:,:,f) - learn_rate .*rot90(conv2(img,rot90(D_CL1(:,:,f),2),'valid'),2);
        end        
        % Gradient descent
        w_h2 = w_h2 - learn_rate .* (delta_o * z_h2');
        b_h2 = b_h2 - learn_rate .* (delta_o * ones(batch_size,1));
        w_h = w_h - learn_rate .* (delta_h2 * z_h');
        b_h = b_h - learn_rate .* (delta_h2 * ones(batch_size,1));
        w_i = w_i - learn_rate .* (delta_h * in);
        b_i = b_i - learn_rate .* (delta_h * ones(batch_size,1));
    end
%     avg_train_err(e) = error(e)/num_train;
 
    % Compute validation error every 5 epoch(for early stopping)
    if(rem(e,5) == 0)
        val_error_sum = 0;
        for i = 1:num_val
            % load img
            img = reshape(X_train(:,i+num_train),D,D);
            % Convolutional layers forward
            % CL1 and SSL 1 
            for f = 1 : num_fm1
                A_CL1(:,:,f) = conv2(img, rot90(W_CL1(:,:,f),2),'valid') + repmat(B_CL1(f),dim_CL1,dim_CL1);
                Z_CL1(:,:,f) = sigmoid(A_CL1(:,:,f));
                [A_SSL1(:,:,f), I_SSL1(:,:,f)]= max_pool(Z_CL1(:,:,f));
            end
            % CL2 and SSL 2 
            for f = 1 : num_fm2
                A_CL2(:,:,f) = conv2(sum(A_SSL1,3), rot90(W_CL2(:,:,f),2),'valid') + repmat(B_CL2(f),dim_CL2,dim_CL2);
                Z_CL2(:,:,f) = sigmoid(A_CL2(:,:,f));
                [A_SSL2(:,:,f), I_SSL2(:,:,f)]= max_pool(Z_CL2(:,:,f));
            end    
            % Fully-connected forward
            in = reshape(A_SSL2,1,num_in);
            sT_train = T_train(i+num_train,:);
            % Forword to hidden layer 1 
            a_h = w_i * in' + repmat(b_i, 1, batch_size);
            z_h = sigmoid(a_h);
            % Forword to hidden layer 2
            a_h2 = w_h * z_h + repmat(b_h, 1, batch_size);
            z_h2 = sigmoid(a_h2);
            % Forword to output layer
            a_out = w_h2 * z_h2 + repmat(b_h2, 1, batch_size);
            delta_o = a_out(1:num_out,:) - sT_train';
            % Compute error
            val_error_sum = val_error_sum + 0.5*sum((a_out' - sT_train).^2);            
        end
%         avg_val_err(e) = val_error_sum / num_val;
        % Compute genralization loss
        GL = 100*(val_error_sum/min_val_error - 1);
        if(val_error_sum <= min_val_error)
            min_val_error = val_error_sum;
            %record_weight
            opt_W_CL1 = W_CL1; opt_B_CL1 = B_CL1; opt_W_CL2 = W_CL2; opt_B_CL2 = B_CL2;
            opt_w_i = w_i; opt_w_h = w_h; opt_w_h2 = w_h2; opt_b_i = b_i; opt_b_h = b_h; opt_b_h2 = b_h2;
        elseif (e > min_epoch && GL > GL_tolerance)
            break;
        end    
    end    
    % decrease learning rate
    learn_rate = fix_learn_rate/(1+0.001*e);
end
%% Testing
load('X_test_white.mat'); load('T_test.mat');
num_test = size(X_test,2);
test_err = 0;
        for i = 1:num_test
            % load img
            img = reshape(X_test(:,i),D,D);
            % Convolutional layers forward
            % CL1 and SSL 1 
            for f = 1 : num_fm1
                A_CL1(:,:,f) = conv2(img, rot90(opt_W_CL1(:,:,f),2),'valid') + repmat(opt_B_CL1(f),dim_CL1,dim_CL1);
                Z_CL1(:,:,f) = sigmoid(A_CL1(:,:,f));
                [A_SSL1(:,:,f), I_SSL1(:,:,f)]= max_pool(Z_CL1(:,:,f));
            end
            % CL2 and SSL 2 
            for f = 1 : num_fm2
                A_CL2(:,:,f) = conv2(sum(A_SSL1,3), rot90(opt_W_CL2(:,:,f),2),'valid') + repmat(opt_B_CL2(f),dim_CL2,dim_CL2);
                Z_CL2(:,:,f) = sigmoid(A_CL2(:,:,f));
                [A_SSL2(:,:,f), I_SSL2(:,:,f)]= max_pool(Z_CL2(:,:,f));
            end    
            % Fully-connected forward
            in = reshape(A_SSL2,1,num_in);
            sT_test = T_test(i,:);
            % Forword to hidden layer 1 
            a_h = opt_w_i * in' + repmat(opt_b_i, 1, batch_size);
            z_h = sigmoid(a_h);
            % Forword to hidden layer 2
            a_h2 = opt_w_h * z_h + repmat(opt_b_h, 1, batch_size);
            z_h2 = sigmoid(a_h2);
            % Forword to output layer
            a_out = opt_w_h2 * z_h2 + repmat(opt_b_h2, 1, batch_size);
            delta_o = a_out(1:num_out,:) - sT_test';
            % Compute error
            test_err = test_err + 0.5*sum((a_out' - sT_test).^2)/num_test;            
        end
display(test_err);