import matplotlib.pyplot as plt
import numpy as np

from hermespy.simulation.simulation import Simulation
from hermespy.modem.modem import Modem
from hermespy.modem.evaluators import BitErrorEvaluator, FrameErrorEvaluator
from hermespy.modem.waveform_generator_psk_qam import WaveformGeneratorPskQam

# Create a new HermesPy simulation scenario
simulation = Simulation()

# Create a new simulated device
device = simulation.new_device()

# Add a modem at the simulated device
modem = Modem()
modem.waveform_generator = WaveformGeneratorPskQam()
modem.device = device

# Monte-Carlo simulation
simulation.add_evaluator(BitErrorEvaluator(modem, modem))
simulation.add_evaluator(FrameErrorEvaluator(modem, modem))
simulation.add_dimension('snr', [10, 4, 2, 1])
simulation.num_samples = 100

result = simulation.run()
result.plot()
plt.show()
