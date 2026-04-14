import matplotlib.pyplot as plt
from hermespy.modem import RootRaisedCosineWaveform

waveform = RootRaisedCosineWaveform(num_preamble_symbols=1, num_data_symbols=0)
waveform.plot_filter()
plt.show()