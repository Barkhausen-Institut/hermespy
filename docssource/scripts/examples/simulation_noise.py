# -*- coding: utf-8 -*-

from hermespy.core import dB, SNRType
from hermespy.simulation import Simulation, AWGN as Noise

# Create a new device
simulation = Simulation()
device = simulation.new_device()

# Configure the device's default noise model
device.noise = Noise()

# Specify the noise's default power
device.noise.power = dB(-20.0)

# Alternatively: Specify the device's receive SNR
device.snr_type = SNRType.PN0
device.snr = dB(20.0)
