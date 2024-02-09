# -*- coding: utf-8 -*-

from hermespy.core import dB

from hermespy.simulation import Simulation, SpecificIsolation
# Create a new device
simulation = Simulation()
device = simulation.new_device()

# Specify the device's isolation with a leakage of -30dB with respect to the
# transmitted signal
device.isolation = SpecificIsolation(dB(-30.0))
