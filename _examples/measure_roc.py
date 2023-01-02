from os import path

import matplotlib.pyplot as plt

from hermespy.core import SNRType
from hermespy.channel import RadarChannel
from hermespy.hardware_loop import HardwareLoop, SimulatedPhysicalScenario
from hermespy.radar import Radar, FMCW, ReceiverOperatingCharacteristic
from hermespy.simulation import SpecificIsolation
from hermespy.tools import db2lin


# Global parameters
bandwidth = 3.072e9
carrier_frequency = 10e9

system = SimulatedPhysicalScenario()
system.snr = 1
system.snr_type = SNRType.N0

hardware_loop = HardwareLoop[SimulatedPhysicalScenario](system)
hardware_loop.num_drops = 2
hardware_loop.results_dir = hardware_loop.default_results_dir()

device = system.new_device(carrier_frequency=carrier_frequency)
device.isolation = SpecificIsolation(db2lin(-80))

radar = Radar()
chirp_duration = 2e-8
radar.waveform = FMCW(bandwidth=bandwidth, num_chirps=10, chirp_duration=chirp_duration, pulse_rep_interval=1.1*chirp_duration)
radar.device = device

channel = RadarChannel(target_range=(.75, 1.25), radar_cross_section=1.)
system.set_channel(device, device, channel)

hardware_loop.run(overwrite=False, campaign='h1_measurements')

channel.target_exists = False
hardware_loop.run(overwrite=False, campaign='h0_measurements')

roc = ReceiverOperatingCharacteristic.From_HDF(path.join(hardware_loop.results_dir, 'drops.h5'))
roc.plot()
plt.show()
