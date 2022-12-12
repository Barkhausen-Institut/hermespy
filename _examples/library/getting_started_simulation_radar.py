"""
This script simulates a monostatic radar detector using FMCW pulses and generates the ROC curve
"""

import matplotlib.pyplot as plt
import numpy as np

# Import required HermesPy modules
from hermespy.simulation import Simulation
from hermespy.radar import Radar, FMCW, ReceiverOperatingCharacteristic, MaxDetector, RootMeanSquareError
from hermespy.channel import RadarChannel
from hermespy.tools.math import db2lin


# Simulation parameters
bandwidth = 100e6
chirp_duration = 1.5e-6
repetition_interval = 3e-6
chirps_in_frame = 10

number_of_drops = 1000

# Create a new HermesPy simulation scenario
simulation = Simulation()

# Create devices and add them to simulation scenario
device_h1 = simulation.scenario.new_device()
device_h1.carrier_frequency = 10e9

device_h0 = simulation.scenario.new_device()
device_h0.carrier_frequency = 10e9

# Configure device operator and associated waveform
operator_h1 = Radar()
operator_h1.waveform = FMCW(bandwidth=bandwidth,
                            chirp_duration=chirp_duration,
                            pulse_rep_interval=repetition_interval,
                            num_chirps=chirps_in_frame)
operator_h1.detector = MaxDetector()
operator_h1.device = device_h1

operator_h0 = Radar()
operator_h0.waveform = FMCW(bandwidth=bandwidth,
                            chirp_duration=chirp_duration,
                            pulse_rep_interval=repetition_interval,
                            num_chirps=chirps_in_frame)
operator_h0.detector = MaxDetector()
operator_h0.device = device_h0

# Creates radar channels and associate them to radar devices:
# channel_h0 is the radar channel without a target, from device_h0 to device_h0
# channel_h1 is the radar channel with a target, from device_h1 to device_h1
# A "blockage channel" is created between device_h0 and device_h1 to ensure that they don't interfere with each other
channel_h1 = RadarChannel(target_range=(operator_h1.waveform.max_range * .1, operator_h1.waveform.max_range * .9), radar_cross_section=1, target_exists=True, attenuate=False)
channel_h0 = RadarChannel(target_range=50, radar_cross_section=1, target_exists=False, attenuate=False)

simulation.scenario.set_channel(device_h1, device_h1, channel_h1)
simulation.scenario.set_channel(device_h0, device_h0, channel_h0)
simulation.scenario.channel(device_h0, device_h1).gain = 0.

# Configure Monte Carlo simulation, it will run over different SNR values for a certain number of drops
simulation.add_evaluator(ReceiverOperatingCharacteristic(receiving_radar=operator_h1,
                                                         receiving_radar_null_hypothesis=operator_h0))
simulation.add_evaluator(RootMeanSquareError(operator_h1, channel_h1))
simulation.num_samples = number_of_drops

snr_db = np.asarray([-15, -12, -9, 0])  # SNR in dB
snr = db2lin(snr_db)

# convert SNR to Ep/N0, which is expected in noise function
EpN0 = snr * chirp_duration * bandwidth

simulation.new_dimension('snr', snr)

# Launch simulation campaign
result = simulation.run()

# Visualize ROC results
result.plot()
result.save_to_matlab("test.mat")
plt.show()
