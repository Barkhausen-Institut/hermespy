import matplotlib.pyplot as plt

from hermespy.core import dB
from hermespy.simulation.simulation import Simulation
from hermespy.modem import TransmittingModem, ReceivingModem, BitErrorEvaluator, ThroughputEvaluator, RootRaisedCosineWaveform
from hermespy.fec import RepetitionEncoder

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
transmitter = TransmittingModem()
transmitter.waveform = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=100, oversampling_factor=8, roll_off=.9)
transmitter.encoder_manager.add_encoder(RepetitionEncoder(repetitions=3))
base_station.transmitters.add(transmitter)

# Configure a receiving modem at the terminal
receiver = ReceivingModem()
receiver.waveform = RootRaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=100, oversampling_factor=8, roll_off=.9)
receiver.encoder_manager.add_encoder(RepetitionEncoder(repetitions=3))
terminal.receivers.add(receiver)

# Configure simulation evaluators
simulation.add_evaluator(BitErrorEvaluator(transmitter, receiver, plot_surface=False))
simulation.add_evaluator(ThroughputEvaluator(transmitter, receiver, plot_surface=True))

# Configure simulation sweep dimensions
snr_dimension = simulation.new_dimension('snr', dB(12, 10, 8, 6, 5, 4, 3, 2, 1, 0))
rep_dimension = simulation.new_dimension('repetitions', [1, 3, 5, 7, 9], transmitter.encoder_manager[0], receiver.encoder_manager[0])
snr_dimension.title = 'SNR'
rep_dimension.title = 'Code Repetitions'

# Run the simulation
simulation.num_samples = 1000
result = simulation.run()

# Plot simulation results
result.plot()
plt.show()
