% nn_es.m
% early stopping version
clear all; clc;
load('X_train.mat'); load('T_train.mat');
% load('X_train_10000.mat'); load('T_train_10000.mat');
X_train = X_train';
num_in = size(X_train,2); 
num_out = 3;
% M : number of hidden unit (excluding bias unit)
% N : training data size
M1 = 300;     M2 = 200;
N = size(X_train,1);
train_ratio = 0.7;
num_train = N * train_ratio;
num_val = N - num_train;
X_val = X_train(num_train+1:end,:); T_val = T_train(num_train+1:end,:);
X_train = X_train(1:num_train,:);  T_train = T_train(1:num_train,:);
batch_size = 10;
num_batch = floor(num_train/batch_size);
% rem_size = rem(N,batch_size);

epoch = 100;
min_epoch = 10;
error = zeros(epoch,1);
val_error = zeros(epoch,1);
min_val_error = Inf;
GL_tolerance = 5;
% learn_rate = 1/sqrt(batch_size)/20;
learn_rate = 0.005;
fix_learn_rate = 0.005;
momentum = 0.5;

% % weight matrices 
w_i = rand_range(-1,1,M1, num_in);     b_i = rand_range(-1,1,M1, 1);
w_h = rand_range(-1,1,M2, M1);         b_h = rand_range(-1,1,M2, 1);
w_h2 = rand_range(-1,1,num_out, M2);   b_h2 = rand_range(-1,1,num_out, 1);
% w_i = normrnd(0,1,M1, num_in);     b_i = normrnd(0,1,M1, 1);
% w_h = normrnd(0,1,M2, M1);         b_h = normrnd(0,1,M2, 1);
% w_h2 = normrnd(0,1,num_out, M2);   b_h2 = normrnd(0,1,num_out, 1);
% w_i = fan_in_out_rand(num_in,M1,[M1, num_in]);     b_i = fan_in_out_rand(1,M1,[M1, 1]);
% w_h = fan_in_out_rand(M1,M2,[M2, M1]);         b_h = fan_in_out_rand(1,M2,[M2, 1]);
% w_h2 = fan_in_out_rand(M2,num_out,[num_out, M2]);   b_h2 = fan_in_out_rand(1,num_out,[num_out, 1]);
% Changes of weights
c_w_i = zeros(M1, num_in);  c_b_i = zeros(M1, 1);
c_w_h = zeros(M2, M1);      c_b_h = zeros(M2, 1);
c_w_h2 = zeros(num_out, M2);c_b_h2 = zeros(num_out, 1);
% optimal weights
opt_w_i = zeros(M1, num_in);  opt_b_i = zeros(M1, 1);
opt_w_h = zeros(M2, M1);      opt_b_h = zeros(M2, 1);
opt_w_h2 = zeros(num_out, M2);opt_b_h2 = zeros(num_out, 1);

%% Training
tic
for e = 1 : epoch
    
    % Shuffle X_train and T_train for new epoch
    order = randperm(size(X_train,1));
    sX_train = X_train(order,:);
    sT_train = T_train(order,:);
    
    for bat = 1 : num_batch
        first = (bat-1)*batch_size + 1;
        last = first + batch_size - 1;
        % Forword to hidden layer 1 
        a_h = w_i * sX_train(first:last,:)' + repmat(b_i, 1, batch_size);
        z_h = sigmoid(a_h);
        % Forword to hidden layer 2
        a_h2 = w_h * z_h + repmat(b_h, 1, batch_size);
        z_h2 = sigmoid(a_h2);
        % Forword to output layer
        a_out = w_h2 * z_h2 + repmat(b_h2, 1, batch_size);
        delta_o = a_out(1:num_out,:) - sT_train(first:last,1:num_out)';
        error(e,1) = error(e,1) + 0.5*sum(sum((a_out' - sT_train(first:last,1:num_out)).^2,1));
        % Backpropagate to hidden layer 2
        dz_da2 = z_h2.*(1.0 - z_h2);
        delta_h2 = dz_da2.*(w_h2' * delta_o);
        % Backpropagate to hidden layer 1
        dz_da = z_h.*(1.0 - z_h);
        delta_h = dz_da.*(w_h' * delta_h2);    
        % Gradient descent
        c_w_h2 = momentum.*c_w_h2 - learn_rate .* (delta_o * z_h2');
        c_b_h2 = momentum.*c_b_h2 - learn_rate .* (delta_o * ones(batch_size,1));        
        c_w_h = momentum.*c_w_h - learn_rate .* (delta_h2 * z_h');
        c_b_h = momentum.*c_b_h - learn_rate .* (delta_h2 * ones(batch_size,1));
        c_w_i = momentum.*c_w_i - learn_rate .* (delta_h * sX_train(first:last,:));
        c_b_i = momentum.*c_b_i - learn_rate .* (delta_h * ones(batch_size,1));
        w_h2 = w_h2 + c_w_h2;
        b_h2 = b_h2 + c_b_h2;
        w_h = w_h + c_w_h;
        b_h = b_h + c_b_h;
        w_i = w_i + c_w_i;
        b_i = b_i + c_b_i;
    end     
    % Compute validation error every 5 epoch(for early stopping)
    val_error(e) = 0;
    if(rem(e,5) == 0)
        % Forword to hidden layer 1 
        a_h = w_i * X_val' + repmat(b_i, 1, num_val);
        z_h = sigmoid(a_h);
        % Forword to hidden layer 2
        a_h2 = w_h * z_h + repmat(b_h, 1, num_val);
        z_h2 = sigmoid(a_h2);
        % Forword to output layer
        a_out = w_h2 * z_h2 + repmat(b_h2, 1, num_val);
        val_error(e) = 0.5*sum(sum((a_out' - T_val).^2,1));
        % Compute Generalization Loss
        GL = 100*(val_error(e)/min_val_error - 1);
        if(val_error(e) <= min_val_error)
            min_val_error = val_error(e);
            %record_weight
            opt_w_i = w_i; opt_w_h = w_h; opt_w_h2 = w_h2; opt_b_i = b_i; opt_b_h = b_h; opt_b_h2 = b_h2;
        elseif (e > min_epoch && GL > GL_tolerance)
            break;
        end  
    end    
    % decrease learning rate
    learn_rate = fix_learn_rate/(1+0.001*e);
end
toc
%% Testing
load('X_test.mat'); load('T_test.mat');
num_test = size(X_test,2);
% Forword to hidden layer 1 
a_h = opt_w_i * X_test + repmat(opt_b_i, 1, num_test);
z_h = sigmoid(a_h);
% Forword to hidden layer 2
a_h2 = opt_w_h * z_h + repmat(opt_b_h, 1, num_test);
z_h2 = sigmoid(a_h2);
% Forword to output layer
a_out = opt_w_h2 * z_h2 + repmat(opt_b_h2, 1, num_test);
test_err = 0.5*sum(sum((a_out' - T_test).^2,1))/num_test;
display(test_err);