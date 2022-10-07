import matplotlib.pyplot as plt

from hermespy.hardware_loop import UsrpSystem
from hermespy.modem import Modem, WaveformGeneratorPskQam, PskQamCorrelationSynchronization, PskQamLeastSquaresChannelEstimation, PskQamZeroForcingChannelEqualization, BitErrorEvaluator


system = UsrpSystem()

device_alpha = system.new_device('192.168.189.132', carrier_frequency=2e9)
device_alpha.tx_gain = 20
device_alpha.rx_gain = 50
device_alpha.max_receive_delay = 3e-6
device_alpha.calibration_delay = 0

oversampling_factor = 4
sampling_rate = device_alpha.max_sampling_rate / oversampling_factor
modem_alpha = Modem()
modem_alpha.waveform_generator = WaveformGeneratorPskQam(oversampling_factor=oversampling_factor, num_preamble_symbols=16, num_data_symbols=100, symbol_rate=sampling_rate, modulation_order=64)
modem_alpha.waveform_generator.synchronization = PskQamCorrelationSynchronization()
modem_alpha.waveform_generator.channel_estimation = PskQamLeastSquaresChannelEstimation()
modem_alpha.waveform_generator.channel_equalization = PskQamZeroForcingChannelEqualization()
modem_alpha.device = device_alpha

tx_signal, tx_symbols, tx_bits = modem_alpha.transmit()
tx_signal.plot('Transmission')

system.trigger()

rx_signal, rx_symbols, rx_bits = modem_alpha.receive()
rx_signal.plot('Reception')
rx_symbols.plot_constellation()
BitErrorEvaluator(modem_alpha, modem_alpha).evaluate().plot()

plt.show()
