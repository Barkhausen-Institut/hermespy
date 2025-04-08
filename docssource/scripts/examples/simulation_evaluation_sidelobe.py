# -*- coding: utf-8 -*-

import numpy as np
from scipy.constants import speed_of_light

from hermespy.beamforming import ConventionalBeamformer
from hermespy.core import AntennaMode, Transformation
from hermespy.simulation import DeviceFocus, SidelobeEvaluator, Simulation, SimulatedUniformArray, SimulatedIdealAntenna, StaticTrajectory
from hermespy.modem import RootRaisedCosineWaveform, SimplexLink


cf = 1e8
simulation = Simulation()

# Initialize a transmitting device featuring 16 ideal isotropic antennas
# arranged in a half-wavelength spaced 4x4 uniform rectangular array.
tx_device = simulation.new_device(
    carrier_frequency=cf,
    pose=StaticTrajectory(Transformation.From_Translation(
        np.array([0, 0, 0])
    )),
    antennas=SimulatedUniformArray(
        SimulatedIdealAntenna,
        0.5 * speed_of_light / cf,
        [4, 4, 1],
    ),
)

# Initialize a receiving device featuring 1 ideal isotropic antenna
# located at 100 meters from the transmitting device in the z-direction.
rx_device = simulation.new_device(
    carrier_frequency=cf,
    pose=StaticTrajectory(Transformation.From_Translation(
        np.array([0, 0, 100])
    )),
)

# Configure a static conventional beamformer for the transmitting device
# with a beam focus at the receiving device.
desired_focus = DeviceFocus(rx_device)
tx_device.transmit_coding[0] = ConventionalBeamformer()
tx_device.transmit_coding[0].transmit_focus = [desired_focus]

# Configure a communication down-link with a transmitting modem at the 
# transmitting device and a receiving modem at the receiving device.
link = SimplexLink(waveform=RootRaisedCosineWaveform())
link.connect(tx_device, rx_device)

# Evaluate the beamforming main lobe level at the transmitting device
simulation.add_evaluator(SidelobeEvaluator(
    tx_device, AntennaMode.TX, desired_focus
))

simulation.num_samples = 10
simulation.run()