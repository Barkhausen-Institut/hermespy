
import matplotlib.pyplot as plt
import numpy as np
#import ray
#ray.init(local_mode=True, num_cpus=1)

from hermespy.core import SNRType
from hermespy.channel import RadarChannel
from hermespy.simulation import AutomaticGainControl, Simulation, SimulatedDevice, SpecificIsolation
from hermespy.radar import Radar, FMCW, ReceiverOperatingCharacteristic
from hermespy.tools import db2lin


# Global parameters
bandwidth = 3.072e9
carrier_frequency = 10e9

# Initialize the base system
simulation = Simulation()
simulation.scenario.snr = 1e-12
simulation.scenario.snr_type = SNRType.N0

device: SimulatedDevice = simulation.new_device(carrier_frequency=carrier_frequency)
device.isolation = SpecificIsolation()
device.adc.gain = AutomaticGainControl()
device.adc.num_quantization_bits = 12

# Configure a root-raised-cosine single carrier communication waveform to be transmitted
radar = Radar()
chirp_duration = 2e-8
radar.waveform = FMCW(bandwidth=bandwidth, num_chirps=10, chirp_duration=chirp_duration, pulse_rep_interval=1.1*chirp_duration)
radar.device = device

channel = RadarChannel(target_range=(.75, 1.25), radar_cross_section=1.)
simulation.scenario.set_channel(device, device, channel)

simulation.add_evaluator(ReceiverOperatingCharacteristic(radar, channel))
simulation.new_dimension('snr', db2lin(np.array([-120, -130])))
simulation.new_dimension('isolation', db2lin(np.array([-80, -70])), simulation.scenario.devices[0].isolation)
simulation.num_samples = 1000

result = simulation.run()
result.plot()
plt.show()
