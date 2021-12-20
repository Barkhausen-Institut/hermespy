import matplotlib.pyplot as plt
import numpy as np

from hermespy import Simulation
from hermespy.modem import Modem, WaveformGeneratorPskQam

# Create a new HermesPy simulation scenario
simulation = Simulation()

# Create a new simulated device
device = simulation.new_device()

# Add a modem at the simulated device
modem = Modem()
modem.waveform_generator = WaveformGeneratorPskQam()
modem.device = device

drop = simulation.drop(40.)
drop.plot_received_symbols()
drop.plot_bit_errors()

drop = simulation.drop(5.)
drop.plot_received_symbols()
drop.plot_bit_errors()

plt.show()

# Monte-Carlo simulation
simulation.noise_loop = 10 ** (np.array([10., 8., 6., 4., 2., 1., 0.1, 1e-2, 1e-3]) / 10)
simulation.max_num_drops = 50
simulation.min_num_drops = 10
simulation.confidence_margin = .8

statistics = simulation.run()
statistics.plot_bit_error_rates()
plt.show()
