import matplotlib.pyplot as plt
from hermespy.modem import RaisedCosineWaveform

waveform = RaisedCosineWaveform(num_preamble_symbols=1, num_data_symbols=0)
waveform.plot_filter_correlation()

plt.show()