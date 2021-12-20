import matplotlib.pyplot as plt
import numpy as np

from hermespy import Scenario, Transmitter, Receiver, Simulation
from hermespy.modem import WaveformGeneratorPskQam

transmitter = Transmitter()
transmitter.waveform_generator = WaveformGeneratorPskQam()

receiver = Receiver()
receiver.waveform_generator = WaveformGeneratorPskQam()

receiver_b = Receiver()
receiver_b.waveform_generator = WaveformGeneratorPskQam()

scenario = Scenario()
scenario.add_transmitter(transmitter)
scenario.add_receiver(receiver)
scenario.add_receiver(receiver_b)


simulation = Simulation()
simulation.add_scenario(scenario)

simulation.noise_loop = 10 ** (np.array([10., 8., 6., 4., 2., 1., 0.1, 1e-2, 1e-3]) / 10)
simulation.max_num_drops = 50
simulation.min_num_drops = 10
simulation.confidence_margin = .8

statistics = simulation.run()
statistics.plot_bit_error_rates()
plt.show()
