% max_pool.m
% Description: subsampling an input squared matrix using 2x2 max pooling
% Input: matrix A
% Ouput: reduced ouput B and the index matrix I of Max values
function [B,I] = max_pool(A)
    % Divide A into 2x2 blocks
    c = mat2cell(A,repmat(2,1,size(A,1)/2),repmat(2,1,size(A,1)/2));
    % Apply 2x2 maxpooling on these blocks
    [Max,i] = cellfun(@max_2x2, c,'UniformOutput', false);
    % Convert cells back to matrices
    B = cell2mat(Max);
    I = cell2mat(i);
end