
from itertools import product
from unittest import TestCase

import numpy as np
from math import cos, sin
from scipy.constants import speed_of_light, pi

from hermespy.core import Signal
from hermespy.channel.cluster_delay_line_rural_macrocells import RuralMacrocellsLineOfSight
from hermespy.simulation import Simulation
from hermespy.core import IdealAntenna, UniformArray

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestClusterDelayLine(TestCase):

    def setUp(self) -> None:

        self.carrier_frequency = 10e9
        self.sampling_rate = 1e3
        self.frequency = .25 * self.sampling_rate

        self.array_dimensions = (30, 30, 1)
        self.antenna_spacing = .5 * speed_of_light / self.carrier_frequency
        self.antennas = UniformArray(IdealAntenna(), self.antenna_spacing, self.array_dimensions)

        self.simulation = Simulation()
        self.device_a = self.simulation.scenario.new_device()
        self.device_b = self.simulation.scenario.new_device()

        self.device_a.antennas = self.antennas
        self.device_b.antennas = UniformArray(IdealAntenna(), self.antenna_spacing, [1, 1, 1])
        self.device_a.position = np.array([0., 0., 0.])
        self.device_b.position = np.array([0., 0., 100.])
        self.device_a.orientation = np.array([0., 0., 0.])
        self.device_b.orientation = np.array([0., 0., 0.])
        self.device_a.carrier_frequency = self.carrier_frequency
        self.device_b.carrier_frequency = self.carrier_frequency

        self.channel = RuralMacrocellsLineOfSight()
        self.simulation.scenario.set_channel(self.device_a, self.device_b, self.channel)
        self.channel.set_seed(123456)

    def test_cdl(self):

        num_samples = 1000
        signal_samples = np.tile(np.exp(2j * pi * self.frequency * np.arange(num_samples) / self.sampling_rate),
                                 (1, 1))
        signal = Signal(signal_samples, sampling_rate=self.sampling_rate, carrier_frequency=self.carrier_frequency)

        reception_a, _, csi = self.channel.propagate(signal)
        samples = reception_a[0].samples

        num_angle_candidates = 80
        zenith_angles = np.linspace(0, pi, num_angle_candidates)
        azimuth_angles = np.linspace(0, 2 * pi, num_angle_candidates)

        dictionary = np.empty((self.antennas.num_antennas, num_angle_candidates ** 2), dtype=complex)
        for i, (aoa, zoa) in enumerate(product(azimuth_angles, zenith_angles)):

            dictionary[:, i] = self.device_a.antennas.spherical_response(self.frequency, aoa, zoa)

        beamformer = np.linalg.norm(dictionary.T @ samples, axis=1, keepdims=False).reshape((num_angle_candidates, num_angle_candidates))
        return
