%% getting_started_example1 script:
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

% instanciation of "Modem"
operator = modem_modules.Modem();

% the function "pyargs" allows matlab to pass parameters in the object
% constructor. In this case, the desired parameter is "oversampling_factor=8", 
% as in the example of https://hermespy.org/getting_started.html.
oversampling_factor = pyargs('oversampling_factor',py.int(8));
% instanciation of "CommunicationWaveformPskQam" with "oversampling_factor"
% parametrization
operator.waveform = modem_modules.CommunicationWaveformPskQam(oversampling_factor);

% some parameters can be set directly. In this case, we set
% "num_preamble_symbols=20", as in the example of https://hermespy.org/getting_started.html
operator.waveform.num_preamble_symbols = py.int(20);

% instanciation of "SimulatedDevice"
operator.device = simulation_modules.SimulatedDevice();

% run function "transmit()", which returns a "tuple" type variable
signal_tuple = operator.transmit(); 

% the desired object in this case is of type "Signal", which is obtained as bellow.
% Obs: typically using the bracket "{n}" returns the desired variable,
% however this method is not guarateed to be sufficient always.
signal = signal_tuple{1};

% we are interested in the samples of a signal "signal.getitem()" which is a ndarray.
% the function "python2matlab" converts ndarray to double
signal_samples = python2matlab(signal.getitem());

% PLOTS
figure
plot(real(signal_samples)); hold on; grid on;
plot(imag(signal_samples));
legend('real','imaginary')


