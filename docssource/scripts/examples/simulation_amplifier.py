# -*- coding: utf-8 -*-

from hermespy.simulation import Simulation, PowerAmplifier

# Create a new device
simulation = Simulation()
device = simulation.new_device()

# Configure the device's default amplification model
device.rf_chain.power_amplifier = PowerAmplifier()
