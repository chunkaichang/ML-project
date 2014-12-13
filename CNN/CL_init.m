% CL_init.m
% initialize matrices used in some convolution layer
% Output: W(weight), B(bias), A(input), Z(output), D(delta), out_size(output size)
% Input: k(receptive field width), f(# of feauture maps), in_size(input size)
function [W,B,A,Z,D,out_size,opt_W, opt_B] = CL_init(k, f, prev_f, in_size)
out_size = in_size - k + 1;
W= fan_in_out_rand(prev_f*k*k,1,[k,k,f]); % weights of CL1
B = zeros(f,1); % biases of CL1
A = zeros(out_size,out_size,f); % inputs of CL1
Z = zeros(out_size,out_size,f); % outputs of CL1
D = zeros(out_size,out_size,f); % deltas of CL1
opt_W= zeros(k,k,f); % optimal weights of CL1 (for early stopping)
opt_B = zeros(f,1); % biases of CL1