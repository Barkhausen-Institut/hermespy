import matplotlib.pyplot as plt

from hermespy.simulation.simulation import Simulation
from hermespy.modem import Modem, BitErrorEvaluator, ThroughputEvaluator, RootRaisedCosineWaveform
from hermespy.coding import RepetitionEncoder

# Create a new HermesPy simulation scenario
simulation = Simulation()

# Create two devices representing base station and terminal
# in a downlink scenario
base_station = simulation.scenario.new_device()
terminal = simulation.scenario.new_device()

# Disable device self-interference by setting the gain 
# of the respective self-inteference channels to zero
simulation.scenario.channel(base_station, base_station).gain = 0.
simulation.scenario.channel(terminal, terminal).gain = 0.

# Configure a transmitting modem at the base station
transmitter = Modem()
transmitter.waveform_generator = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=100, oversampling_factor=8, roll_off=.9)
transmitter.device = base_station
transmitter.encoder_manager.add_encoder(RepetitionEncoder(repetitions=3))

# Configure a receiving modem at the terminal
receiver = Modem()
receiver.waveform_generator = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=100, oversampling_factor=8, roll_off=.9)
receiver.device = terminal
receiver.encoder_manager.add_encoder(RepetitionEncoder(repetitions=3))

# Configure simulation evaluators
simulation.add_evaluator(BitErrorEvaluator(transmitter, receiver))
simulation.add_evaluator(ThroughputEvaluator(transmitter, receiver))

# Configure simulation sweep dimensions
snr_dimension = simulation.new_dimension('snr', [10, 8, 6, 4, 2, 1, 0.5, 0.25, .125, .0625])
rep_dimension = simulation.new_dimension('repetitions', [1, 3, 5, 7, 9], transmitter.encoder_manager[0], receiver.encoder_manager[0])
snr_dimension.title = 'SNR'
rep_dimension.title = 'Code Repetitions'

# Run the simulation
simulation.num_samples = 1000
result = simulation.run()

# Plot simulation results
result.plot()
plt.show()
