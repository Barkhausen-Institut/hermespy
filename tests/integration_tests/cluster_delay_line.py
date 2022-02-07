
from itertools import product
from unittest import TestCase

import numpy as np
from math import cos, sin
from scipy.constants import speed_of_light, pi

from hermespy.core.signal_model import Signal
from hermespy.channel.cluster_delay_line_templates import RuralMacrocellsLineOfSight
from hermespy.simulation import Simulation


class TestClusterDelayLine(TestCase):

    def setUp(self) -> None:

        self.carrier_frequency = 10e9
        self.sampling_rate = 100e6
        self.frequency = .25 * self.sampling_rate

        num_antennas_x = 10
        num_antennas_y = 1
        num_antennas_z = 1
        self.num_antennas = num_antennas_x * num_antennas_y * num_antennas_z
        topology = np.empty((self.num_antennas, 3), dtype=float)

        for i, (z, y, x) in enumerate(product(range(num_antennas_z), range(num_antennas_y), range(num_antennas_x))):
            topology[i, :] = (x, y, z)

        topology *= speed_of_light / self.carrier_frequency

        self.simulation = Simulation()
        self.device_a = self.simulation.new_device()
        self.device_b = self.simulation.new_device()

        self.device_a.topology = topology
        self.device_b.topology = topology
        self.device_a.position = np.array([0., 0., 0.])
        self.device_b.position = np.array([0., 0., 100.])
        self.device_a.orientation = np.array([0., 0., 0.])
        self.device_b.orientation = np.array([0., pi, 0.])
        self.device_a.carrier_frequency = self.carrier_frequency
        self.device_b.carrier_frequency = self.carrier_frequency

        self.channel = RuralMacrocellsLineOfSight()
        self.simulation.set_channel(self.device_a, self.device_b, self.channel)

        return

    def test_cdl(self):

        num_samples = 1000
        signal_samples = np.tile(np.exp(2j * pi * self.frequency * np.arange(num_samples) / self.sampling_rate), (self.num_antennas, 1))
        signal = Signal(signal_samples, sampling_rate=self.sampling_rate, carrier_frequency=self.carrier_frequency)

        reception_a, _, csi = self.channel.propagate(signal)
        samples = reception_a[0].samples

        num_angle_candidates = 25
        theta_angles = np.linspace(0, pi, num_angle_candidates)
        phi_angles = np.linspace(-pi, pi, num_angle_candidates)

        dictionary = np.empty((self.num_antennas, num_angle_candidates ** 2), dtype=complex)
        for i, (theta, phi) in enumerate(product(theta_angles, phi_angles)):

            wave_vector = -2j * pi * speed_of_light / self.device_a.carrier_frequency * np.array([sin(theta)*cos(phi),
                                                                                         sin(theta)*sin(phi),
                                                                                         cos(theta)])
            dictionary[:, i] = np.exp(np.inner(wave_vector, self.device_a.topology))

        beamformer = dictionary.T @ samples

        return
