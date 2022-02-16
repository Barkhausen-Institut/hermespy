import matplotlib.pyplot as plt

# Import required HermesPy modules
from hermespy.channel import Channel
from hermespy.simulation import Simulation
from hermespy.modem import Modem, WaveformGeneratorPskQam, BitErrorEvaluator

# Create a new HermesPy simulation scenario
simulation = Simulation()

# Create devices
tx_device = simulation.new_device()
rx_device = simulation.new_device()

# Configure device operators
tx_operator = Modem()
tx_operator.waveform_generator = WaveformGeneratorPskQam(oversampling_factor=8)
tx_operator.device = tx_device

rx_operator = Modem()
rx_operator.waveform_generator = WaveformGeneratorPskQam(oversampling_factor=8)
rx_operator.device = rx_device

# Configure the channel model between the two simulated devices
simulation.set_channel(rx_device, tx_device, Channel())
simulation.set_channel(rx_device, rx_device, None)
simulation.set_channel(tx_device, tx_device, None)

# Configure Monte Carlo simulation
simulation.add_evaluator(BitErrorEvaluator(tx_operator, tx_operator))
simulation.add_evaluator(BitErrorEvaluator(rx_operator, rx_operator))
simulation.add_dimension('snr', [10, 4, 2, 1, 0.5])
simulation.num_samples = 1000

# Launch simulation campaign
result = simulation.run()

# Visualize results
result.plot()
plt.show()
