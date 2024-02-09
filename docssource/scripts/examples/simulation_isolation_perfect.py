# -*- coding: utf-8 -*-

from hermespy.simulation import Simulation, PerfectIsolation

# Create a new device
simulation = Simulation()
device = simulation.new_device()

# Specify the device's isolation model
device.isolation = PerfectIsolation()
