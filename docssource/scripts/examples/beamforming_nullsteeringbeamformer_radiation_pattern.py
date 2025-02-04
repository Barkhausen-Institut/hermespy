from scipy.constants import pi,speed_of_light

carrier_frequency = 72e9
wavelength = speed_of_light / carrier_frequency
half_wavelength = wavelength / 2

# Configure a uniform antenna array 
from hermespy.simulation import SimulatedIdealAntenna, SimulatedUniformArray

uniform_array = SimulatedUniformArray(SimulatedIdealAntenna, half_wavelength, (8, 8, 1))

# Configure beamformer
from hermespy.beamforming.nullsteeringbeamformer import NullSteeringBeamformer
from hermespy.beamforming import SphericalFocus

beamformer = NullSteeringBeamformer()
beamformer.transmit_focus = [SphericalFocus(0,0),SphericalFocus(-.5*pi,.25*pi),SphericalFocus(.5*pi,.25*pi)]

# Render the beamformer's characteristics
import matplotlib.pyplot as plt

_ = uniform_array.plot_pattern(carrier_frequency, beamformer, title="Null Steering Beamformer")
plt.show()