import matplotlib.pyplot as plt

from hermespy.simulation import SimulatedDevice
from hermespy.modem import TransmittingModem, RaisedCosineWaveform

device = SimulatedDevice()
transmitter = TransmittingModem()
waveform = RaisedCosineWaveform(modulation_order=16, oversampling_factor=16, num_preamble_symbols=0, symbol_rate=1e8, num_data_symbols=1000, roll_off=.9)
transmitter.waveform = waveform
device.add_dsp(transmitter)

device.transmit().mixed_signal.eye(symbol_duration=1/waveform.symbol_rate, domain='complex')
plt.show()