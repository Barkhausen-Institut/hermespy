
import matplotlib.pyplot as plt
import numpy as np
import ray
# ray.init(local_mode=True, num_cpus=1)

from hermespy.channel import RadarChannel
from hermespy.simulation import Simulation
from hermespy.radar import Radar, FMCW, ReceiverOperatingCharacteristic


# Global parameters
bandwidth = 3.072e9
carrier_frequency = 10e9

# Initialize the base system
simulation = Simulation()
device = simulation.new_device(carrier_frequency=carrier_frequency)

# Configure a root-raised-cosine single carrier communication waveform to be transmitted
radar = Radar()
chirp_duration = 2e-8
radar.waveform = FMCW(bandwidth=bandwidth, num_chirps=10, chirp_duration=chirp_duration, pulse_rep_interval=1.1*chirp_duration)
radar.device = device

channel = RadarChannel(target_range=(0, radar.waveform.max_range), radar_cross_section=1., attenuate=False)
simulation.scenario.set_channel(device, device, channel)

simulation.add_evaluator(ReceiverOperatingCharacteristic(radar, channel))
simulation.new_dimension('snr', np.linspace(1e-2, 1, 10, endpoint=True))
simulation.num_samples = 5000

result = simulation.run()
result.plot()
plt.show()
