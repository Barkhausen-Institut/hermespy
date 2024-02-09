# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import speed_of_light

from hermespy.core import AntennaMode, Transformation
from hermespy.simulation import SimulatedCustomArray, Simulation, SimulatedPatchAntenna, SimulatedUniformArray, SimulatedIdealAntenna, SimulatedAntennaPort


carrier_frequency = 3.7e9
wavelength = speed_of_light / carrier_frequency
antennas = SimulatedCustomArray()
for x in range(10):
    antennas.add_antenna(SimulatedPatchAntenna(
        pose=Transformation.From_Translation(np.array([.5*x*wavelength, 0, 0]))
    ))

antennas.plot_topology()
antennas.plot_pattern(carrier_frequency, AntennaMode.TX)

uniform_array = SimulatedUniformArray(SimulatedIdealAntenna, .5 * wavelength, [10, 1, 1])

uniform_array.plot_topology()

hybrid_array = SimulatedCustomArray()
for x in range(10):
    port = SimulatedAntennaPort(
        pose=Transformation.From_Translation(np.array([.5*x*wavelength, 0, 0]))
    )
    for y in range(5):
        port.add_antenna(SimulatedPatchAntenna(
            pose=Transformation.From_Translation(np.array([0, .5*y*wavelength, 0]))
        ))
    hybrid_array.add_port(port)

hybrid_array.plot_topology()

simulation = Simulation()
device = simulation.new_device()
device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, .5 * wavelength, [10, 1, 1])

plt.show()
