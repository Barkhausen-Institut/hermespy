import numpy as np

from hermespy.core import SignalExtractor, Signal, SignalTransmitter, SignalReceiver
from hermespy.channel import IdealChannel
from hermespy.simulation import Simulation, RfChain, AnalogDigitalConverter, OscillatorPhaseNoise




carrier_frequency = 1e9
sampling_rate = 1e8
adc_bits_no = 16

simulation = Simulation(num_samples=1000)

tx_device = simulation.new_device(
    carrier_frequency=carrier_frequency,
    sampling_rate=sampling_rate,
)
tx_device.rf_chain = RfChain()
tx_device.rf_chain.adc = AnalogDigitalConverter(
    num_quantization_bits=adc_bits_no,
)
tx_device.rf_chain.phase_noise = OscillatorPhaseNoise()

rx_device = simulation.new_device(
    carrier_frequency=carrier_frequency,
    sampling_rate=sampling_rate,
)
rx_device.rf_chain = RfChain()
rx_device.rf_chain.adc = AnalogDigitalConverter(
    num_quantization_bits=adc_bits_no,
)
rx_device.rf_chain.phase_noise = OscillatorPhaseNoise()

channel = IdealChannel()
simulation.set_channel(tx_device, rx_device, channel)

data_iq = np.random.standard_normal(100000) + 1j * np.random.standard_normal(100000)

signal_transmitter = SignalTransmitter(
    Signal.Create(
        data_iq,
        sampling_rate=sampling_rate,
        carrier_frequency=carrier_frequency,
    )
)
tx_device.add_dsp(signal_transmitter)
signal_receiver = SignalReceiver(
    len(data_iq), sampling_rate=sampling_rate
)
rx_device.add_dsp(signal_receiver)

# Extract signals
simulation.add_evaluator(SignalExtractor(signal_transmitter))
simulation.add_evaluator(SignalExtractor(signal_receiver))

result = simulation.run()
result.save_to_matlab("rri_reds_result.mat")
