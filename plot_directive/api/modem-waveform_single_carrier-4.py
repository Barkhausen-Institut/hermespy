import matplotlib.pyplot as plt

from hermespy.modem import RootRaisedCosineWaveform


waveform = RootRaisedCosineWaveform(oversampling_factor=16, symbol_rate=1e6, num_preamble_symbols=1, num_data_symbols=0)
waveform.plot_filter_correlation()
plt.show()