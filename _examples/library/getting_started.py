import matplotlib.pyplot as plt
from hermespy import Scenario, Transmitter
from hermespy.modem import WaveformGeneratorPskQam

transmitter = Transmitter()
transmitter.waveform_generator = WaveformGeneratorPskQam()

scenario = Scenario()
scenario.add_transmitter(transmitter)

signal, _ = transmitter.send()
signal.plot()

transmitter.waveform_generator.num_preamble_symbols = 0
signal, _ = transmitter.send()
signal.plot()

plt.show()
