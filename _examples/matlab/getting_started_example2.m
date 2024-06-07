%% getting_started_example2 script:
%
% DESCRIPTION:
%   Example on how to use the hermespy (https://hermespy.org/)  functions
%   in matlab. In particular, this scrip shows how to run on example of 
%   https://hermespy.org/getting_started.html. For detailed information
%   about the hermes functions, please search directly in the referred website.
%   
%
% requirements:
%   - hermes should be installed, see https://hermespy.org/installation.html
%   - python enviroment should be set, example: pyenv('Version','<your folder>\<your hermes env>\python')
%   - tested matlab version: MATLAB R2020b
%   - tested HermesPy version: 0.3.0
%
% Author: Roberto Bomfin
% Date: 04/04/2022
%%
close all

% import modules
simulation_modules = py.importlib.import_module('hermespy.simulation');
modem_modules = py.importlib.import_module('hermespy.modem');
channel_modules = py.importlib.import_module('hermespy.channel');

% instanciation of SimulatedDevice objects
tx_device = simulation_modules.SimulatedDevice(); 
rx_device = simulation_modules.SimulatedDevice();

% instanciation of Modem
tx_operator = modem_modules.Modem();
% the function "pyargs" allows matlab to pass parameters in the object
% constructor. In this case, the desired parameter is "oversampling_factor=8", 
% as in the example of https://hermespy.org/getting_started.html.
oversampling_factor = pyargs('oversampling_factor',py.int(8));
% instanciation of "CommunicationWaveformPskQam" with "oversampling_factor"
% parametrization
tx_operator.waveform = modem_modules.CommunicationWaveformPskQam(oversampling_factor);
% set tx_device
tx_operator.device = tx_device;

% analogous to tx_device
rx_operator = modem_modules.Modem();
rx_operator.waveform = modem_modules.CommunicationWaveformPskQam(oversampling_factor);
rx_operator.device = rx_device;

% Simulate a channel between the two devices
channel = channel_modules.Channel(tx_operator.device, rx_operator.device);

% Simulate the signal transmission over the channel
transmit = tx_operator.transmit(); 

% the desired object in this case is of type "Signal", which is obtained as bellow.
% Obs: typically using the bracket "{n}" returns the desired variable,
% however this method is not guarateed to be sufficient always.
tx_signal = transmit{1};

% run function "propagate", which returns a tuple type variable
propagate = channel.propagate(tx_signal); 

% returns "list" type variable where:
% - rx_signal is a "list" variable and is in the position {1}
% - channel_state is a "ChannelStateInformation" variable and is in the position {3}
% original python sintaxe for completeness: 
%   rx_signal, _, channel_state = channel.propagate(tx_signal)
rx_signal = propagate{1};
channel_state = propagate{3};

% define SNR
snr = pyargs('noise_level',py.float(5));
rx_device.receive(rx_signal,snr);

% function "receive()" returns a "tuple" type variable, where
% - rx_symbols is a "Symbols" type variable and is in position {2}
% - rx_bits is a ndarray type variable and is in the position {3}
% original python sintaxe for completeness:
%       _, rx_symbols, rx_bits = rx_operator.receive()
receive = rx_operator.receive(); 
rx_symbols = receive{2}; 
rx_bits = receive{3};

% Evaluate bit errors during transmission and 
evaluator = modem_modules.BitErrorEvaluator(tx_operator, rx_operator);
eval = evaluator.evaluate();

% convert variables from python to matlab
ber = python2matlab(eval.artifact);
symbols = python2matlab(rx_symbols.raw);

disp(['ber = ' num2str(mean(ber))])

% visualize the received symbol constellation
figure
plot(real(symbols),imag(symbols),'o')


