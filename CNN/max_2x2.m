% max_2x2.m
% Description: return the max value in a 2x2 matrix and its index(row,col)
function [Max,index] = max_2x2(A)
    [m,i1]=max(A);
    [Max,i2] = max(m);
    row = i1(i2); 
    col = i2;
    index = (col-1)*2 + row;