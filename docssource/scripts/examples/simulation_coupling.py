# -*- coding: utf-8 -*-

from hermespy.simulation import Simulation, PerfectCoupling as Coupling


# Create a new device
simulation = Simulation()
device = simulation.new_device()

# Configure a custom analog-digital conversion model
device.coupling = Coupling()
