import matplotlib.pyplot as plt

from hermespy.hardware_loop import UsrpSystem
from hermespy.modem import DuplexModem, RootRaisedCosineWaveform, SingleCarrierCorrelationSynchronization, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization, BitErrorEvaluator


system = UsrpSystem()

device_alpha = system.new_device('192.168.189.133', carrier_frequency=1e9)
device_alpha.tx_gain = 30
device_alpha.rx_gain = 30
device_alpha.max_receive_delay = 5e-6
device_alpha.calibration_delay = 0
device_alpha.adaptive_sampling = False

oversampling_factor = 4
sampling_rate = device_alpha.max_sampling_rate / oversampling_factor
modem_alpha = DuplexModem()
modem_alpha.waveform_generator = RootRaisedCosineWaveform(oversampling_factor=oversampling_factor, num_preamble_symbols=16,
                                                          num_data_symbols=100, symbol_rate=sampling_rate, modulation_order=16)
modem_alpha.waveform_generator.synchronization = SingleCarrierCorrelationSynchronization()
modem_alpha.waveform_generator.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
modem_alpha.waveform_generator.channel_equalization = SingleCarrierZeroForcingChannelEqualization()
modem_alpha.device = device_alpha

system.drop()

modem_alpha.transmission.signal.plot()
modem_alpha.reception.signal.plot('Reception')
modem_alpha.reception.equalized_symbols.plot_constellation()

BitErrorEvaluator(modem_alpha, modem_alpha).evaluate().plot()

plt.show()
