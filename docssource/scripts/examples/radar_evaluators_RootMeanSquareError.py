import matplotlib.pyplot as plt

from hermespy.core import dB
from hermespy.radar import Radar, FMCW, MaxDetector, RootMeanSquareError
from hermespy.simulation import Simulation
from hermespy.channel import SingleTargetRadarChannel

# Create a new simulated scenario featuring a single device
simulation = Simulation(num_samples=1000)
device = simulation.new_device(carrier_frequency=60e9)

# Configure the device to transmit and reveive radar waveforms
radar = Radar(device=device)
radar.waveform = FMCW()
radar.detector = MaxDetector()

# Create a new radar channel with a single illuminated target
target = SingleTargetRadarChannel((1, radar.max_range), 1., attenuate=False)
simulation.scenario.set_channel(device, device, target)

# Create a new detection probability evaluator
simulation.add_evaluator(RootMeanSquareError(radar, radar, target))

# Sweep over the target's SNR during the simulation
simulation.new_dimension('noise_level', dB(0, -5, -10, -20, -30), device)

# Run the simulation
result = simulation.run()
result.plot()
plt.show()

