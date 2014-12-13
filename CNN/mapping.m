% mapping.m
% Description: Set the element in A{index} to a input value
% Input: matrix A, val, index
function o = mapping(A,val,index)
    A(index)= val;
    o = A;
end