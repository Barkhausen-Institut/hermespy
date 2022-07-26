import matplotlib.pyplot as plt

# Import required HermesPy modules
from hermespy.channel import Channel
from hermespy.simulation import Simulation
from hermespy.modem import Modem, RootRaisedCosineWaveform, BitErrorEvaluator

# Create a new HermesPy simulation scenario
simulation = Simulation()

# Create device
device = simulation.scenario.new_device()

# Configure device operator
operator = Modem()
operator.waveform_generator = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=40, oversampling_factor=8, roll_off=.9)
operator.device = device

# Configure Monte Carlo simulation
simulation.add_evaluator(BitErrorEvaluator(operator, operator))
simulation.new_dimension('snr', [10, 4, 2, 1, 0.5])
simulation.num_samples = 1000

# Launch simulation campaign
result = simulation.run()

# Visualize results
result.plot()
plt.show()
