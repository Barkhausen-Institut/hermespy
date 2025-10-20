import matplotlib.pyplot as plt
import numpy as np

# Import required HermesPy modules
from hermespy.core import dB, Transformation
from hermespy.channel import IndoorOffice, LOSState
from hermespy.simulation import LinearTrajectory, Simulation, StaticTrajectory, SNR
from hermespy.modem import BitErrorEvaluator, RootRaisedCosineWaveform, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization, SimplexLink

# Create a new HermesPy simulation scenario
simulation = Simulation(seed=42, num_samples=1000)

# Create two devices representing base station and terminal
# in a downlink scenario
cf = 2.4e9
os = 8
bw = 1e6
base_station = simulation.scenario.new_device(carrier_frequency=cf, bandwidth=bw, oversampling_factor=os)
terminal = simulation.scenario.new_device(carrier_frequency=cf, bandwidth=bw, oversampling_factor=os)

# Assign a positions / trajectories to the terminal and base station
base_station.trajectory = StaticTrajectory(
    Transformation.From_Translation(np.array([0, 0, 2]))
)
terminal.trajectory = LinearTrajectory(
    Transformation.From_Translation(np.array([10, 0, 0])),
    Transformation.From_Translation(np.array([0, 10, 0])),
    10,
)

# Configure a downlink communicating between base station and terminal
# via a single-carrier waveform
link = SimplexLink()
link.waveform = RootRaisedCosineWaveform(
    num_preamble_symbols=10, num_data_symbols=100, roll_off=.9,
)
link.waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
link.waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()
base_station.transmitters.add(link)
terminal.receivers.add(link)

# Specify the channel model between base station and terminal
channel = IndoorOffice(expected_state=LOSState.LOS)
simulation.set_channel(base_station, terminal, channel)

# Specify the hardware noise model
base_station.noise_level = SNR(dB(100), base_station, channel)
terminal.noise_level = SNR(dB(100), base_station, channel)

# Evaluate the bit error rate
ber = BitErrorEvaluator(link, link)
simulation.add_evaluator(ber)

# Run a simulation generating drops every 1 second
simulation.drop_interval = 1

# Sweep over the receive SNR
simulation.new_dimension('noise_level', dB(100, 20, 16, 12, 8, 4, 0), terminal)
result = simulation.run()
result.plot()
plt.show()
