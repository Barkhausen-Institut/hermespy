import matplotlib.pyplot as plt
from hermespy.modem import RaisedCosineWaveform

waveform = RaisedCosineWaveform(oversampling_factor=16, symbol_rate=1e6, num_preamble_symbols=1, num_data_symbols=0)
waveform.plot_filter()
plt.show()