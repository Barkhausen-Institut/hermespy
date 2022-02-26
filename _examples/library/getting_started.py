import matplotlib.pyplot as plt

from hermespy.simulation import SimulatedDevice
from hermespy.modem import Modem, WaveformGeneratorPskQam

operator = Modem()
operator.waveform_generator = WaveformGeneratorPskQam(oversampling_factor=8)
operator.device = SimulatedDevice()

signal, _, _ = operator.transmit()

signal.plot()
plt.show()
