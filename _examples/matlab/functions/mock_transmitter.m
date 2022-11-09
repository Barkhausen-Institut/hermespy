%% mock_transmitter function:
%
% DESCRIPTION:
%   - This function creats a Mock transmitter object with with necessary
%   properties
%   - Extension of new properties should be
%     relatively easy to implement by the user if needed for further
%     applications by checking new input parameters and assigning a default
%     value if field is not provided.
%
% INPUT:
%   input - struct with input parameters
%
% OUTPUT:
%   output - mock transmitter object
%
%
% Author: Roberto Bomfin
% Date: 28/06/2022

function transmitter = mock_transmitter(input)

% check input arguments
if isfield(input,'carrier_frequency')
    carrier_frequency = input.carrier_frequency;
else
    carrier_frequency = 2.4e9; % default value
end

if isfield(input,'velocity')
    velocity = input.velocity;
else
    velocity = 0; % default value
end

if isfield(input,'sampling_rate')
    sampling_rate = input.sampling_rate;
else
    sampling_rate = 10000; % default value
end

spherical_response = py.unittest.mock.Mock(pyargs('return_value',py.numpy.array(py.numpy.complex128([1.]))));
args_antennas = pyargs('num_antennas',py.int(1),'spherical_response',spherical_response);
antennas = py.unittest.mock.Mock(args_antennas);

velocity_array = py.numpy.array({0.,0.,1/2 * velocity});

tx_args = pyargs(...
    'carrier_frequency',py.float(carrier_frequency),...
    'sampling_rate',py.float(sampling_rate),...
    'antennas',antennas, ...
    'velocity',velocity_array);
transmitter = py.unittest.mock.Mock(tx_args);

end