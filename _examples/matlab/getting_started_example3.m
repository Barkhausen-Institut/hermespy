%% getting_started_example3 script:
%
% DESCRIPTION:
%   Example on how to use the hermespy (https://hermespy.org/)  functions
%   in matlab. In particular, this scrip shows how to use the radar_channel
%   module of hermes in matlab.
%   For detailed information about the hermes functions, please search 
%   directly in the referred website (https://hermespy.org/).
%   
%
% requirements:
%   - hermes should be installed, see https://hermespy.org/installation.html
%   - python enviroment should be set, example: pyenv('Version','<your folder>\<your hermes env>\python')
%   - tested matlab version: MATLAB R2020b
%   - tested HermesPy version: 0.3.0
%
% Author: Roberto Bomfin
% Date: 28/06/2022
%%
close all

% import modules
channel_modules = py.importlib.import_module('hermespy.channel');
core = py.importlib.import_module('hermespy.core');

% parameters
sampling_rate = 1.0e6;
carrier_frequency = 1.0e9;
velocity = 300;

input_tx_mock.carrier_frequency = carrier_frequency;
input_tx_mock.sampling_rate = sampling_rate;
input_tx_mock.velocity = velocity;

transmitter = mock_transmitter(input_tx_mock);
receiver = transmitter;

% instanciate radar channel
range = py.float(10);
radar_cross_section = py.float(1);
target_exists = py.bool(1);
losses_db = py.float(0);

radar_args = pyargs(...
    'target_range',range,...
    'radar_cross_section',radar_cross_section,...
    'transmitter',transmitter,...
    'receiver',receiver,...
    'target_exists',target_exists,...
    'losses_db',losses_db, ...
    'target_velocity',velocity);

radar_channel = channel_modules.RadarChannel(radar_args);

% create input signal
num_samples = 1000;
sinewave_frequency = 0 * sampling_rate;
time = (0:num_samples-1) / sampling_rate;
input_signal = cos(2 * pi * sinewave_frequency * time);

% create python ndarray
input_signal_py = matlab2python(input_signal,'complex64');

% resize is necessary to match expected input shape of radar_channel
input_signal_py.resize(py.int(1),py.int(num_samples)); 

% create a Signal object
signal = core.Signal.Create(input_signal_py, py.int(transmitter.sampling_rate));

% propate input through the channel
output_tupple = radar_channel.propagate(signal);

% retrieve the out samples
output_list = output_tupple{1};
output_cell = output_list.cell{1};
output = python2matlab(output_cell[:, :]);

% plot received signal after the radar channel
figure
plot(real(output)); hold on;
plot(imag(output))

