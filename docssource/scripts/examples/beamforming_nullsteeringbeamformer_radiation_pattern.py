import matplotlib.pyplot as plt
from scipy.constants import pi,speed_of_light
import numpy as np
from itertools import product

from hermespy.simulation import SimulatedIdealAntenna, SimulatedCustomArray
from hermespy.beamforming.nullsteeringbeamformer import NullSteeringBeamformer
from hermespy.core import AntennaMode,Transformation
from hermespy.beamforming import SphericalFocus


carrier_frequency = 72e9
wavelength = speed_of_light / carrier_frequency

# Configure an antenna array with custom coordinates
uniform_array = SimulatedCustomArray()
for x, y in product(range(4), range(4)):
    uniform_array.add_antenna(SimulatedIdealAntenna(
        mode=AntennaMode.TX,
        pose=Transformation.From_Translation(np.array([.5 * wavelength * x, .5 * wavelength * y, 0])),
    ))

# Configure beamformer
beamformer = NullSteeringBeamformer()
beamformer.transmit_focus = [SphericalFocus(0,0),SphericalFocus(-.5*pi,.25*pi),SphericalFocus(.5*pi,.25*pi)]

# Render the beamformer's characteristics
_ = uniform_array.plot_pattern(carrier_frequency, beamformer, title="Null Steering Beamformer")
plt.show()