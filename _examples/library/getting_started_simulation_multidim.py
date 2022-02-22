import matplotlib.pyplot as plt

from hermespy.simulation.simulation import Simulation
from hermespy.modem.modem import Modem
from hermespy.modem.evaluators import BitErrorEvaluator, ThroughputEvaluator
from hermespy.modem.waveform_generator_psk_qam import WaveformGeneratorPskQam
from hermespy.coding import RepetitionEncoder

# Create a new HermesPy simulation scenario
simulation = Simulation()

# Create a new simulated device
device = simulation.scenario.new_device()

# Add a modem at the simulated device
modem = Modem()
modem.waveform_generator = WaveformGeneratorPskQam()
modem.device = device
modem.encoder_manager.add_encoder(RepetitionEncoder(repetitions=3))

# Configure simulation evaluators
simulation.add_evaluator(BitErrorEvaluator(modem, modem))
simulation.add_evaluator(ThroughputEvaluator(modem, modem))

# Configure simulation sweep dimensions
snr_dimension = simulation.new_dimension('snr', [4, 2, 1, 0.5, 0.25, .125])
rep_dimension = simulation.new_dimension('repetitions', [1, 3, 5, 7], modem.encoder_manager[0])
snr_dimension.title = 'SNR'
rep_dimension.title = 'Code Repetitions'

# Run the simulation
simulation.num_samples = 1000
result = simulation.run()

# Plot simulation results
result.plot()
plt.show()
