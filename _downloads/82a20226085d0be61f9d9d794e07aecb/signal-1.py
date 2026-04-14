import matplotlib.pyplot as plt

from hermespy.simulation import SimulatedDevice
from hermespy.modem import TransmittingModem, RaisedCosineWaveform

device = SimulatedDevice()
transmitter = TransmittingModem()
waveform = RaisedCosineWaveform(modulation_order=16, num_preamble_symbols=0, num_data_symbols=1000, roll_off=.9)
transmitter.waveform = waveform
device.add_dsp(transmitter)

device.transmit().mixed_signal.eye(domain='time')
plt.show()