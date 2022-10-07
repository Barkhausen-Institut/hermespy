import matplotlib.pyplot as plt

from hermespy.simulation import SimulatedDevice
from hermespy.modem import DuplexModem, RootRaisedCosineWaveform

operator = DuplexModem()
operator.waveform_generator = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=40, oversampling_factor=8, roll_off=.9)
operator.device = SimulatedDevice()

transmission = operator.transmit()

transmission.signal.plot()
plt.show()
