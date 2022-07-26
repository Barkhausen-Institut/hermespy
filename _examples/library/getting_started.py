import matplotlib.pyplot as plt

from hermespy.simulation import SimulatedDevice
from hermespy.modem import Modem, RootRaisedCosineWaveform

operator = Modem()
operator.waveform_generator = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=40, oversampling_factor=8, roll_off=.9)
operator.device = SimulatedDevice()

signal, _, _ = operator.transmit()

signal.plot()
plt.show()
