% SSL_backprop.m
% Description: backpropagate from SSL to CL

function O = SSL_backprop(D_CL,D_SSL,I_SSL)
    % divide delta of CL into 2x2 blocks
    c_D_CL = mat2cell(D_CL,repmat(2,1,size(D_CL,1)/2),repmat(2,1,size(D_CL,1)/2));
    c_D_SSL = num2cell(D_SSL);
    c_I_SSL = num2cell(I_SSL);
    % mapping back delta of SSL to the positioin in CL where max value is contributed 
    c_D_CL = cellfun(@mapping,c_D_CL,c_D_SSL,c_I_SSL,'UniformOutput', false);
    % Convert cells back to matrix
    O = cell2mat(c_D_CL);
end