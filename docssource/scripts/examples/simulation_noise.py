# -*- coding: utf-8 -*-

from hermespy.core import dB
from hermespy.simulation import Simulation, AWGN as NoiseModel, N0 as NoiseLevel, SNR

# Create a new device
simulation = Simulation()
device = simulation.new_device()

# Configure the device's default noise model
device.noise_model = NoiseModel()

# Specify the noise's default power
device.noise_level = NoiseLevel(dB(-20))

# Alternatively: Specify the device's receive SNR
device.noise_level = SNR(dB(20), device)

# Shorthand for the noise level
device.noise_level << dB(-21)
