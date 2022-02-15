import matplotlib.pyplot as plt

from hermespy.channel import Channel
from hermespy.simulation.simulation import Simulation
from hermespy.modem.modem import Modem
from hermespy.modem.evaluators import BitErrorEvaluator
from hermespy.modem.waveform_generator_psk_qam import WaveformGeneratorPskQam

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

# Monte-Carlo simulation
simulation.add_evaluator(BitErrorEvaluator(tx_operator, rx_operator))
simulation.add_dimension('snr', [10, 4, 2, 1, 0.5])
simulation.num_samples = 1000
result = simulation.run()

result.plot()
plt.show()
