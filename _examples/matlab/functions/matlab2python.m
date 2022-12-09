%% matlab2python function:
%
% DESCRIPTION:
%   - This function converts matlab vector of double to python ndarray
%   - This function is still under development and does not cover
%     all data types of python. However, extension to new data types should be 
%     relatively easy to implement by the user.
%
% INPUT:
%   input - matlab row vector of size 1 x N
%   type - currently supported 'complex64' and 'complex128'
%
% OUTPUT:
%   output - python ndarray of shape (N,)
%
%
% Author: Roberto Bomfin
% Date: 28/06/2022

function [output] = matlab2python(input,type)

switch type
    case 'complex128'
        output = py.numpy.fromstring(num2str(input), ....
            py.numpy.complex128, int8(-1), char(' '));
    case 'complex64'
        output = py.numpy.fromstring(num2str(input), ....
            py.numpy.complex64, int8(-1), char(' '));    
    otherwise        
        disp(['Type not supported: ' type]);
end

end