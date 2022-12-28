"""
This script simulates a monostatic radar detector using FMCW pulses
"""
from operator import attrgetter
import matplotlib.pyplot as plt

from hermespy.channel import RadarChannel
from hermespy.radar import Radar, FMCW
from hermespy.simulation import SimulatedDevice
from hermespy.tools import db2lin


# Simulation parameters
chirp_duration = 1.5e-6
pulse_repetition_interval = 3e-6
chirp_bandwidth = 500e6
adc_sampling_rate = None
carrier_frequency = 60e9
num_chirps = 10

target_velocity = 0.
target_range = 200
snr = db2lin(0)

# Create two simulated devices acting as monostatic radars
# one will be simulated with a target (H1) and one with no targets (H0)
device_h1 = SimulatedDevice(carrier_frequency=carrier_frequency)
device_h0 = SimulatedDevice(carrier_frequency=carrier_frequency)

# Define a radar operation and an associated waveform on both devices
operator_h1 = Radar()
operator_h1.waveform = FMCW(chirp_duration=chirp_duration,
                            bandwidth=chirp_bandwidth,
                            num_chirps=num_chirps,
                            sampling_rate=2*chirp_bandwidth,
                            pulse_rep_interval=pulse_repetition_interval,
                            adc_sampling_rate=adc_sampling_rate)
operator_h1.device = device_h1

operator_h0 = Radar()
operator_h0.waveform = FMCW(chirp_duration=chirp_duration,
                            bandwidth=chirp_bandwidth,
                            num_chirps=num_chirps,
                            sampling_rate=2*chirp_bandwidth,
                            pulse_rep_interval=pulse_repetition_interval,
                            adc_sampling_rate=adc_sampling_rate)
operator_h0.device = device_h0

# Simulate a radar channel for each device, one with (H1) and one without (H0) a target
channel_h1 = RadarChannel(target_range=target_range, velocity=target_velocity, radar_cross_section=1, target_exists=True, attenuate=False)
channel_h1.transmitter = device_h1
channel_h1.receiver = device_h1

channel_h0 = RadarChannel(target_range=target_range, velocity=target_velocity, radar_cross_section=1, target_exists=False, attenuate=False)
channel_h0.transmitter = device_h0
channel_h0.receiver = device_h0

# Simulate the signal transmission over the channel
operator_h1.transmit()
rx_signal_h1, _, channel_state_h1 = channel_h1.propagate(device_h1.transmit())

operator_h0.transmit().signal
rx_signal_h0, _, channel_state_h0 = channel_h0.propagate(device_h0.transmit())

# Simulate the signal reception and radar processing
device_h1.process_input(rx_signal_h1, snr=snr)
operator_h1.receive()

device_h0.process_input(rx_signal_h0, snr=snr)
operator_h0.receive()

# Plot range profile
operator_h1.cube.plot_range('Range Profile with Target')
operator_h0.cube.plot_range('Range Profile without Target')

# Plot range-Doppler map
operator_h1.cube.plot_range_velocity()

plt.show()
