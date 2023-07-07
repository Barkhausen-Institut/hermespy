import matplotlib.pyplot as plt

from hermespy.modem import TransmittingModem, RaisedCosineWaveform

transmitter = TransmittingModem()
waveform = RaisedCosineWaveform(modulation_order=16, oversampling_factor=16, num_preamble_symbols=0, symbol_rate=1e8, num_data_symbols=1000, roll_off=.9)
transmitter.waveform_generator = waveform

transmitter.transmit().signal.plot_eye(1 / waveform.symbol_rate, domain='complex')
plt.show()