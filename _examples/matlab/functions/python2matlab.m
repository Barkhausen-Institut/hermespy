%% python2matlab function:
%
% DESCRIPTION:
%   - This function converts python arrays such as ndarray to double
%   - This function is still under development and does not cover all data 
%     types of python. However, extension to new data types should be 
%     relatively easy to implement by the user.
%
% INPUT:
%   input - (python type data), currently ndarray and tuple has been tested
%
% OUTPUT:
%   output - (double) converted variable from python to matlab
%
%
% Author: Roberto Bomfin
% Date: 04/04/2022

function [output] = python2matlab(input)

switch class(input)
    case 'py.numpy.ndarray'
        if input.ndim.double == 1
            output = cell2mat(cell(input.tolist));
        else
            output = cell2mat(cell(input.tolist{1}));
        end
    case 'py.tuple'
        output =   cell2mat(cell(input{1}.tolist));
    otherwise
        data_class = class(input);
        disp(['Data class not supported: ' data_class]);
end

end