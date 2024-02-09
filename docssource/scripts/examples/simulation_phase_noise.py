# -*- coding: utf-8 -*-

from hermespy.simulation import Simulation, NoPhaseNoise as PhaseNoise

# Create a new device
simulation = Simulation()
device = simulation.new_device()

# Configure the device's default phase noise model
device.rf_chain.phase_noise = PhaseNoise()
