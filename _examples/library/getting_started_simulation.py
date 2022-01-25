import matplotlib.pyplot as plt
import numpy as np

from hermespy.simulation.simulation import Simulation
from hermespy.modem.modem import Modem
from hermespy.modem.evaluators import BitErrorEvaluator
from hermespy.modem.waveform_generator_psk_qam import WaveformGeneratorPskQam

# Create a new HermesPy simulation scenario
simulation = Simulation()

# Create a new simulated device
device = simulation.new_device()

# Add a modem at the simulated device
modem = Modem()
modem.waveform_generator = WaveformGeneratorPskQam()
modem.device = device

#drop = simulation.drop(40.)
#drop.plot_received_symbols()
#drop.plot_bit_errors()
#
#drop = simulation.drop(5.)
#drop.plot_received_symbols()
#drop.plot_bit_errors()
#
#plt.show()

# Monte-Carlo simulation
simulation.add_evaluator(BitErrorEvaluator(modem, modem))
simulation.add_dimension('snr', [10, 4, 2, 1, 0.5])
simulation.num_samples = 1000
#simulation.add_dimension()

result = simulation.run()
result.plot()
plt.show()
