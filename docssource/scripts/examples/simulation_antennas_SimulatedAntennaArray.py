# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import Transformation
from hermespy.simulation import Simulation, SimulatedCustomArray, SimulatedIdealAntenna, SimulatedUniformArray

# Class alias for SimulatedCustomArray
SimulatedAntennaArray = SimulatedCustomArray

simulation = Simulation()
device = simulation.new_device(carrier_frequency=1e8)
device.antennas = SimulatedAntennaArray()


uniform_array = SimulatedUniformArray(SimulatedIdealAntenna, 1e-2, [2, 2, 1])

custom_array = SimulatedCustomArray()
for x, y in np.ndindex((2, 2)):
    custom_array.add_antenna(SimulatedIdealAntenna(
        pose=Transformation.From_Translation(np.array([x*1e-2, y*1e-2, 0])),
    ))

simulation.new_dimension('antennas', [uniform_array, custom_array], device)

simulation.num_drops = 1
result = simulation.run()
result.print()
