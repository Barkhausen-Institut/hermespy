# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from hermespy.beamforming import ConventionalBeamformer
from hermespy.core import AntennaMode
from hermespy.channel import SingleTargetRadarChannel
from hermespy.simulation.rf.presets.ti.xwr1843 import TIXWR1843
from hermespy.simulation import SimulationScenario, SimulatedPatchAntenna, SimulatedCustomArray
from hermespy.radar import Radar

carrier_frequency = 77.5e9
target_range = 4  # m
target_velocity = -10  # m/s


# Initialize a simulation scenario with a single device
scenario = SimulationScenario()
device = scenario.new_device(
    carrier_frequency=carrier_frequency,
    bandwidth=3e9,
    oversampling_factor=2
)

# Configure the device with the XWR1843 RF chain preset
rf = TIXWR1843()
device.rf = rf

# Configure the device antennas as a 3x4 patch antenna array
device.antennas = SimulatedCustomArray()
for _ in range(3):
    device.antennas.add_antenna(SimulatedPatchAntenna(AntennaMode.TX))
for _ in range(4):
    device.antennas.add_antenna(SimulatedPatchAntenna(AntennaMode.RX))

# Configure a radar DSP algorithm
device.add_dsp(Radar(
    waveform=rf,
    receive_beamformer=ConventionalBeamformer(),
))

# Configure a monstatic radar self-interference channel
channel = SingleTargetRadarChannel(
    target_range=target_range,
    velocity=target_velocity,
    radar_cross_section=1.0,
    attenuate=True,
)
scenario.set_channel(device, device, channel)

# Generate a single drop
drop = scenario.drop()
transmission = drop.device_transmissions[0].mixed_signal
reception = drop.device_receptions[0].impinging_signals[0]

#drop.device_receptions[0].impinging_signals[0].plot(title="RF Rx")
drop.device_transmissions[0].mixed_signal.plot(title="RF Tx")
drop.device_receptions[0].operator_inputs[0].plot(title="BB Rx")
drop.device_receptions[0].operator_receptions[0].cube.plot_range()
drop.device_receptions[0].operator_receptions[0].cube.plot_range_velocity()
plt.show()
