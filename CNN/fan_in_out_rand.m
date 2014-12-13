% fan_in_out_rand.m
% initialization of weights in NN according to fan-in and fan-out
function M = fan_in_out_rand(in,out,d)
    upper = 4 * sqrt(6/(in+out));
    lower = -upper;
    M = (upper-lower).*rand(d) + lower;
