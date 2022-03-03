# -*- coding: utf-8 -*-
"""
=============================
3GPP Cluster Delay Line Model
=============================
"""

from __future__ import annotations
from abc import abstractmethod
from math import atan, ceil, sin, cos, sqrt
from typing import Any, List

import numpy as np
from scipy.constants import pi, speed_of_light

from hermespy.core.factory import Serializable
from hermespy.tools.math import db2lin, rotation_matrix
from hermespy.tools.resampling import delay_resampling_matrix
from .channel import Channel

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ClusterDelayLineBase(Channel):

    # Cluster scaling factors for the angle of arrival
    __azimuth_scaling_factors = np.array([[4, .779],
                                          [5, .86],
                                          [8, 1.018],
                                          [10, 1.090],
                                          [11, 1.123],
                                          [12, 1.146],
                                          [14, 1.19],
                                          [15, 1.211],
                                          [16, 1.226],
                                          [19, 1.273],
                                          [20, 1.289]], dtype=float)

    __zenith_scaling_factors = np.array([[8, .889],
                                         [10, .957],
                                         [11, 1.031],
                                         [12, 1.104],
                                         [15, 1.108],
                                         [19, 1.184],
                                         [20, 1.178]], dtype=float)

    # Ray offset angles
    __ray_offset_angles = np.array([.0447, -.0447, .1413, -.1413, .2492, -.2492, .3715, -.3715, .5129, -.5129,
                                    .6797, -.6797, .8844, -.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551])

    # Sub-cluster partitions for the three strongest clusters
    __subcluster_indices: List[List[int]] = [[0, 1, 2, 3, 4, 5, 6, 7, 18, 19],
                                             [8, 9, 10, 11, 16, 17],
                                             [12, 13, 14, 15]]

    def __init__(self, **kwargs) -> None:

        Channel.__init__(self, **kwargs)

    @property
    @abstractmethod
    def line_of_sight(self) -> bool:
        """Does this model assume direct line of sight between the two devices?

        Referred to as :math:`LOS` within the standard.

        Returns:
            bool: Line of sight indicator.
        """
        ...

    @property
    @abstractmethod
    def delay_spread_mean(self) -> float:
        """Mean of the cluster delay spread.

        The spread realization and its mean are referred to as
        :math:`\\mathrm{DS}` and :math:`\\mu_{\\mathrm{lgDS}}` within the the standard, respectively.

        Returns:
            float: Mean delay spread in seconds.
        """
        ...

    @property
    @abstractmethod
    def delay_spread_std(self) -> float:
        """Standard deviation of the cluster delay spread.

        The spread realization and its standard deviation are referred to as
        :math:`\\mathrm{DS}` and :math:`\\sigma_{\\mathrm{lgDS}}` within the the standard, respectively.

        Returns:
            float: Delay spread standard deviation in seconds.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...

    @property
    @abstractmethod
    def aod_spread_mean(self) -> float:
        """Mean of the Azimuth Angle-of-Departure spread.

        The spread realization and its mean are referred to as
        :math:`\\mathrm{ASD}` and :math:`\\mu_{\\mathrm{lgASD}}` within the the standard, respectively.

        Returns:
            float: Mean angle spread in seconds
        """
        ...

    @property
    @abstractmethod
    def aod_spread_std(self) -> float:
        """Standard deviation of the Azimuth Angle-of-Departure spread.

        The spread realization and its standard deviation are referred to as
        :math:`\\mathrm{ASD}` and :math:`\\sigma_{\\mathrm{lgASD}}` within the the standard, respectively.

        Returns:
            float: Angle spread standard deviation in seconds.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...

    @property
    @abstractmethod
    def aoa_spread_mean(self) -> float:
        """Mean of the Azimuth Angle-of-Arriaval spread.

        The spread realization and its mean are referred to as
        :math:`\\mathrm{ASA}` and :math:`\\mu_{\\mathrm{lgASA}}` within the the standard, respectively.

        Returns:
            float: Mean angle spread in seconds
        """
        ...

    @property
    @abstractmethod
    def aoa_spread_std(self) -> float:
        """Standard deviation of the Azimuth Angle-of-Arrival spread.

        The spread realization and its standard deviation are referred to as
        :math:`\\mathrm{ASA}` and :math:`\\sigma_{\\mathrm{lgASA}}` within the the standard, respectively.

        Returns:
            float: Angle spread standard deviation in seconds.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...

    @property
    @abstractmethod
    def zoa_spread_mean(self) -> float:
        """Mean of the Zenith Angle-of-Arriaval spread.

        The spread realization and its mean are referred to as
        :math:`\\mathrm{ZSA}` and :math:`\\mu_{\\mathrm{lgZSA}}` within the the standard, respectively.

        Returns:
            float: Mean angle spread in seconds
        """
        ...

    @property
    @abstractmethod
    def zoa_spread_std(self) -> float:
        """Standard deviation of the Zenith Angle-of-Arrival spread.

        The spread realization and its standard deviation are referred to as
        :math:`\\mathrm{ZSA}` and :math:`\\sigma_{\\mathrm{lgZSA}}` within the the standard, respectively.

        Returns:
            float: Angle spread standard deviation in seconds.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...
    
    @property
    @abstractmethod
    def zod_spread_mean(self) -> float:
        """Mean of the Zenith Angle-of-Departure spread.

        The spread realization and its mean are referred to as
        :math:`\\mathrm{ZOD}` and :math:`\\mu_{\\mathrm{lgZSD}}` within the the standard, respectively.

        Returns:
            float: Mean angle spread in degrees
        """
        ...

    @property
    @abstractmethod
    def zod_spread_std(self) -> float:
        """Standard deviation of the Zenith Angle-of-Departure spread.

        The spread realization and its standard deviation are referred to as
        :math:`\\mathrm{ZOD}` and :math:`\\sigma_{\\mathrm{lgZOD}}` within the the standard, respectively.

        Returns:
            float: Angle spread standard deviation in degrees.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...

    @property
    @abstractmethod
    def zod_offset(self) -> float:
        """Offset between Zenith Angle-of-Arrival and Angle-of-Departure.

        The offset is referred to as :math:`\\mu_{\\mathrm{offset,ZOD}}` within the standard.

        Returns:
            float: The offset in degrees.
        """
        ...

    ###############################
    # ToDo: Shadow fading function
    ###############################

    @property
    @abstractmethod
    def rice_factor_mean(self) -> float:
        """Mean of the rice factor distribution.

        The rice factor realization and its mean are referred to as
        :math:`K` and :math:`\\mu_K` within the the standard, respectively.

        Returns:
            float: Rice factor mean in dB.
        """
        ...

    @property
    @abstractmethod
    def rice_factor_std(self) -> float:
        """Standard deviation of the rice factor distribution.

        The rice factor realization and its standard deviation are referred to as
        :math:`K` and :math:`\\sigma_K` within the the standard, respectively.

        Returns:
            float: Rice factor standard deviation in dB.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...

    @property
    @abstractmethod
    def delay_scaling(self) -> float:
        """Delay scaling proportionality factor

        Referred to as :math:`r_{\\tau}` within the standard.

        Returns:
            float: Scaling factor.

        Raises:
            ValueError:
                If scaling factor is smaller than one.
        """
        ...

    @property
    @abstractmethod
    def cross_polarization_power_mean(self) -> float:
        """Mean of the cross-polarization power.

        The cross-polarization power and its mean are referred to as
        :math:`\\mathrm{XPR}` and :math:`\\mu_{\\mathrm{XPR}}` within the the standard, respectively.

        Returns:
            float: Mean power in dB.
        """
        ...

    @property
    @abstractmethod
    def cross_polarization_power_std(self) -> float:
        """Standard deviation of the cross-polarization power.

        The cross-polarization power and its standard deviation are referred to as
        :math:`\\mathrm{XPR}` and :math:`\\sigma_{\\mathrm{XPR}}` within the the standard, respectively.

        Returns:
            float: Power standard deviation in dB.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...

    @property
    @abstractmethod
    def num_clusters(self) -> int:
        """Number of clusters.

        Referred to as :math:`M` within the standard.

        Returns:
            int: Number of clusters.

        Raises:
            ValueError: If the number of clusters is smaller than one.
        """
        ...

    @property
    @abstractmethod
    def num_rays(self) -> int:
        """Number of rays per cluster.

        Referred to as :math:`N` within the standard.

        Returns:
            int: Number of rays.

        Raises:
            ValueError: If the number of clusters is smaller than one.
        """
        ...

    @property
    @abstractmethod
    def cluster_delay_spread(self) -> float:
        """Delay spread within an individual cluster.

        Referred to as :math:`c_{DS}` within the standard.

        Returns:
            float: Delay spread in seconds.

        Raises:
            ValueError: If spread is smaller than zero.
        """
        ...

    @property
    @abstractmethod
    def cluster_aod_spread(self) -> float:
        """Azimuth Angle-of-Departure spread within an individual cluster.

        Referred to as :math:`c_{ASD}` within the standard.

        Returns:
            float: Angle spread in degrees.

        Raises:
            ValueError: If spread is smaller than zero.
        """
        ...

    @property
    @abstractmethod
    def cluster_aoa_spread(self) -> float:
        """Azimuth Angle-of-Arrival spread within an individual cluster.

        Referred to as :math:`c_{ASA}` within the standard.

        Returns:
            float: Angle spread in degrees.

        Raises:
            ValueError: If spread is smaller than zero.
        """
        ...

    @property
    @abstractmethod
    def cluster_zoa_spread(self) -> float:
        """Zenith Angle-of-Arrival spread within an individual cluster.

        Referred to as :math:`c_{ZSA}` within the standard.

        Returns:
            float: Angle spread in degrees.

        Raises:
            ValueError: If spread is smaller than zero.
        """
        ...

    @property
    @abstractmethod
    def cluster_shadowing_std(self) -> float:
        """Standard deviation of the cluster shadowing.

        Referred to as :math:`\\zeta` within the the standard.

        Returns:
            float: Cluster shadowing standard deviation.

        Raises:
            ValueError: If the deviation is smaller than zero.
        """
        ...

    def _cluster_delays(self,
                        delay_spread: float,
                        rice_factor: float) -> np.ndarray:
        """Compute a single sample set of normalized cluster delays.

        A single cluster delay is referred to as :math:`\\tau_n` within the the standard.

        Args:

            delay_spread (float):
                Delay spread in seconds.

            rice_factor (float):
                Rice factor K in dB.

        Returns:

            np.ndarray:
                Vector of cluster delays.
        """

        delays = - self.delay_scaling * delay_spread * np.log(self._rng.uniform(size=self.num_clusters))

        delays -= delays.min()
        delays.sort()

        # In case of line of sight, scale the delays by the appropriate K-factor
        if self.line_of_sight:
            rice_scale = .775 - .0433 * rice_factor + 2e-4 * rice_factor ** 2 + 17e-6 * rice_factor ** 3
            delays /= rice_scale

        return delays

    def _cluster_powers(self,
                        delay_spread: float,
                        delays: np.ndarray,
                        rice_factor: float) -> np.ndarray:
        """Compute a single sample set of normalized cluster power factors from delays.

        A single cluster power factor is referred to as :math:`P_n` within the the standard.

        Args:

            delay_spread (float):
                Delay spread in seconds.

            delays (np.ndarray):
                Vector of cluster delays.

            rice_factor (float):
                Rice factor K in dB.

        Returns:

            np.ndarray:
                Vector of cluster power scales.
        """

        shadowing = 10 ** (-.1 * self._rng.normal(scale=self.cluster_shadowing_std, size=delays.shape))
        powers = np.exp(-delays * (self.delay_scaling - 1) / (self.delay_scaling * delay_spread)) * shadowing

        # In case of line of sight, add a specular component to the cluster delays
        if self.line_of_sight:

            linear_rice_factor = db2lin(rice_factor)
            powers /= (1 + linear_rice_factor) * np.sum(powers.flat)
            powers[0] += linear_rice_factor / (1 + linear_rice_factor)

        else:
            powers /= np.sum(powers.flat)

        return powers

    def _ray_azimuth_angles(self,
                            cluster_powers: np.ndarray,
                            rice_factor: float,
                            los_azimuth: float) -> np.ndarray:
        """Compute cluster ray azimuth angles of arrival or departure.

        Args:

            cluster_powers (np.ndarray):
                Vector of cluster powers. The length determines the number of clusters.

            rice_factor (float):
                Rice factor in dB.

            los_azimuth (float):
                Line of sight azimuth angle to the target in degrees.

        Returns:
            np.ndarray:
                Matrix of angles in degrees.
                The first dimension indicates the cluster index, the second dimension the ray index.
        """

        # Determine the closest scaling factor
        scale_index = np.argmin(np.abs(self.__azimuth_scaling_factors[:, 0] - len(cluster_powers)))
        angle_scale = self.__azimuth_scaling_factors[scale_index, 1]
        size = cluster_powers.shape

        # Scale the scale (hehe) in the line of sight case
        if self.line_of_sight:
            angle_scale *= 1.1035 - .028 * rice_factor - 2e-3 * rice_factor ** 2 + 1e-4 * rice_factor ** 3

        # Draw azimuth angle spread from the distribution
        spread = self._rng.lognormal(self.aoa_spread_mean, self.aoa_spread_std, size=size)

        angles = 2 * (spread / 1.4) * np.sqrt(-np.log(cluster_powers / cluster_powers.max())) / angle_scale

        # Assign positive / negative integers and add some noise
        angle_variation = self._rng.normal(0., (spread / 7) ** 2, size=size)
        angle_spread_sign = self._rng.choice([-1., 1.], size=size)
        angles: np.ndarray = angle_spread_sign * angles + angle_variation

        # Add the actual line of sight term
        if self.line_of_sight:

            # The first angle within the list is exactly the line of sight component
            angles += los_azimuth - angles[0]

        else:

            angles += los_azimuth

        # Spread the angles
        ray_offsets = self.cluster_aoa_spread * self.__ray_offset_angles
        ray_angles = np.tile(angles[:, None], len(ray_offsets)) + ray_offsets

        return ray_angles

    def _ray_zoa(self,
                 cluster_powers: np.ndarray,
                 rice_factor: float,
                 los_zenith: float) -> np.ndarray:
        """Compute cluster ray zenith angles of arrival.

        Args:

            cluster_powers (np.ndarray):
                Vector of cluster powers. The length determines the number of clusters.

            rice_factor (float):
                Rice factor in dB.

            los_zenith (float):
                Line of sight zenith angle to the target in degrees.

        Returns:
            np.ndarray:
                Matrix of angles in degrees.
                The first dimension indicates the cluster index, the second dimension the ray index.
        """

        size = cluster_powers.shape

        # Select the scaling factor
        scale_index = np.argmin(np.abs(self.__zenith_scaling_factors[:, 0] - len(cluster_powers)))
        zenith_scale = self.__zenith_scaling_factors[scale_index, 1]

        if self.line_of_sight:
            zenith_scale *= 1.3086 + .0339 * rice_factor - .0077 * rice_factor ** 2 + 2e-4 * rice_factor ** 3

        # Draw zenith angle spread from the distribution
        zenith_spread = self._rng.lognormal(self.zoa_spread_mean, self.zoa_spread_std, size=size)

        # Generate angle starting point
        cluster_zenith = -zenith_spread * np.log(cluster_powers / cluster_powers.max()) / zenith_scale

        cluster_variation = self._rng.normal(0., (zenith_spread / 7) ** 2, size=size)
        cluster_sign = self._rng.choice([-1., 1.], size=size)

        # ToDo: Treat the BST-UT case!!!! (los_zenith = 90°)
        cluster_zenith: np.ndarray = cluster_sign * cluster_zenith + cluster_variation

        if self.line_of_sight:
            cluster_zenith += los_zenith - cluster_zenith[0]

        else:
            cluster_zenith += los_zenith

        # Spread the angles
        ray_offsets = self.cluster_zoa_spread * self.__ray_offset_angles
        ray_zenith = np.tile(cluster_zenith[:, None], len(ray_offsets)) + ray_offsets

        return ray_zenith

    def _ray_zod(self,
                 cluster_powers: np.ndarray,
                 rice_factor: float,
                 los_zenith: float) -> np.ndarray:
        """Compute cluster ray zenith angles of departure.

        Args:

            cluster_powers (np.ndarray):
                Vector of cluster powers. The length determines the number of clusters.

            rice_factor (float):
                Rice factor in dB.

            los_zenith (float):
                Line of sight zenith angle to the target in degrees.

        Returns:
            np.ndarray:
                Matrix of angles in degrees.
                The first dimension indicates the cluster index, the second dimension the ray index.
        """

        size = cluster_powers.shape

        # Select the scaling factor
        scale_index = np.argmin(np.abs(self.__zenith_scaling_factors[:, 0] - len(cluster_powers)))
        zenith_scale = self.__zenith_scaling_factors[scale_index, 1]

        if self.line_of_sight:
            zenith_scale *= 1.3086 + .0339 * rice_factor - .0077 * rice_factor ** 2 + 2e-4 * rice_factor ** 3

        # Draw zenith angle spread from the distribution
        zenith_spread = self._rng.lognormal(self.zod_spread_mean, self.zod_spread_std, size=size)

        # Generate angle starting point
        cluster_zenith = -zenith_spread * np.log(cluster_powers / cluster_powers.max()) / zenith_scale

        cluster_variation = self._rng.normal(0., (zenith_spread / 7) ** 2, size=size)
        cluster_sign = self._rng.choice([-1., 1.], size=size)

        # ToDo: Treat the BST-UT case!!!! (los_zenith = 90°)
        # Equation 7.5-19
        cluster_zenith: np.ndarray = cluster_sign * cluster_zenith + cluster_variation + self.zod_offset

        if self.line_of_sight:
            cluster_zenith += los_zenith - cluster_zenith[0]

        else:
            cluster_zenith += los_zenith

        # Spread the angles
        # Equation 7.5 -20
        ray_offsets = 3 / 8 * 10 ** self.zoa_spread_mean * self.__ray_offset_angles
        ray_zenith = np.tile(cluster_zenith[:, None], len(ray_offsets)) + ray_offsets

        return ray_zenith


    def impulse_response(self,
                         num_samples: int,
                         sampling_rate: float) -> np.ndarray:

        center_frequency = self.transmitter.carrier_frequency

        delay_spread = self._rng.lognormal(self.delay_spread_mean, self.delay_spread_std)
        rice_factor = self._rng.normal(loc=self.rice_factor_mean, scale=self.rice_factor_std)

        # Query device positions and orientations
        tx_position = self.transmitter.position
        rx_position = self.receiver.position

        # Positions may not be unspecified
        if tx_position is None or rx_position is None:
            raise ValueError("Cluster delay line models require specified transmitter and receiver positions")

        # Compute the respective angles of arrival and departure
        # Note: We assume that we may substitute zenith by elevation in this model
        los_vector = rx_position - tx_position
        tx_los_vector = rotation_matrix(self.transmitter.orientation).T @ los_vector
        rx_los_vector = rotation_matrix(self.receiver.orientation).T @ los_vector

        los_aoa = atan(rx_los_vector[1] / rx_los_vector[0]) if rx_los_vector[1] != 0. and rx_los_vector[0] != 0. else 0.
        los_aod = atan(tx_los_vector[1] / tx_los_vector[0]) if tx_los_vector[1] != 0. and tx_los_vector[0] != 0. else 0
        los_zoa = atan(sqrt(rx_los_vector[0] ** 2 + rx_los_vector[1] ** 2) / rx_los_vector[2]) if rx_los_vector[2] != 0. else .5 * pi
        los_zod = atan(sqrt(tx_los_vector[0] ** 2 + tx_los_vector[1] ** 2) / tx_los_vector[2]) if tx_los_vector[2] != 0. else .5 * pi

        num_clusters = self.num_clusters
        num_rays = 20

        cluster_delays = self._cluster_delays(delay_spread, rice_factor)
        cluster_powers = self._cluster_powers(delay_spread, cluster_delays, rice_factor)

        ray_aod = pi / 180 * self._ray_azimuth_angles(cluster_powers, rice_factor, 180 * los_aod / pi)
        ray_aoa = pi / 180 * self._ray_azimuth_angles(cluster_powers, rice_factor, 180 * los_aoa / pi)
        ray_zod = pi / 180 * self._ray_zod(cluster_powers, rice_factor, 180 * los_zod / pi)  # ToDo: Zenith departure modeling
        ray_zoa = pi / 180 * self._ray_zoa(cluster_powers, rice_factor, 180 * los_zoa / pi)

        # ToDo: Couple cluster angles randomly

        # Generate cross-polarization power ratios (step 9)
        xpr = 10 ** (.1 * self._rng.normal(self.cross_polarization_power_mean,
                                           self.cross_polarization_power_std,
                                           size=(num_clusters, num_rays)))

        # Draw initial random phases (step 10)
        jones_matrix = np.exp(2j * pi * self._rng.uniform(size=(2, 2, num_clusters, num_rays)))
        jones_matrix[0, 1, ::] *= xpr ** -.5
        jones_matrix[1, 0, ::] *= xpr ** -.5

        # Initialize channel matrices
        num_delay_samples = 1 + ceil(cluster_delays.max() * sampling_rate)
        impulse_response = np.ndarray((num_samples, self.receiver.antennas.num_antennas,
                                       self.transmitter.antennas.num_antennas, num_delay_samples), dtype=complex)

        # Compute the number of clusters, considering the first two clusters get split into 3 partitions
        num_split_clusters = min(2, num_clusters)
        virtual_num_clusters = 3 * num_split_clusters + max(0, num_clusters - 2)

        # Prepare the channel coefficient storage
        nlos_coefficients = np.zeros((virtual_num_clusters, num_samples, self.receiver.antennas.num_antennas,
                                      self.transmitter.antennas.num_antennas), dtype=complex)

        # Prepare the cluster delays, equation 7.5-26
        subcluster_delays = (np.repeat(cluster_delays[:num_split_clusters, None], 3, axis=1) +
                             self.cluster_delay_spread * np.array([1., 1.28, 2.56]))
        virtual_cluster_delays = np.concatenate((subcluster_delays.flatten(), cluster_delays[num_split_clusters:]))

        # Wavelength factor
        wavelength_factor = 2j * pi * self.transmitter.carrier_frequency / speed_of_light
        relative_velocity = self.receiver.velocity - self.transmitter.velocity
        fast_fading = wavelength_factor * np.arange(num_samples) / sampling_rate

        for subcluster_idx in range(0, virtual_num_clusters):

            cluster_idx = int(subcluster_idx / 3) if subcluster_idx < 6 else subcluster_idx - 4
            ray_indices = self.__subcluster_indices[cluster_idx] if cluster_idx < num_split_clusters else range(num_rays)

            for aoa, zoa, aod, zod, jones in zip(ray_aoa[cluster_idx, ray_indices], ray_zoa[cluster_idx, ray_indices],
                                                 ray_aod[cluster_idx, ray_indices], ray_zod[cluster_idx, ray_indices],
                                                 jones_matrix[:, :, cluster_idx, ray_indices].transpose(2, 0, 1)):

                # Equation 7.5-23
                rx_response = self.receiver.antennas.spherical_response(center_frequency, aoa, zoa)

                # Equation 7.5-24
                tx_response = self.transmitter.antennas.spherical_response(center_frequency, aod, zod)

                # Equation 7.5-28
                rx_polarization = self.receiver.antennas.polarization(aoa, zoa)
                tx_polarization = self.transmitter.antennas.polarization(aod, zod)

                channel = ((rx_response[:, None] * rx_polarization) @ jones @ (tx_polarization * tx_response[:, None]).T
                           * sqrt(cluster_powers[cluster_idx] / num_clusters))

                wave_vector = np.array([cos(aoa) * sin(zoa), sin(aoa) * sin(zoa), cos(zoa)], dtype=float)
                impulse = np.exp(2j * pi * center_frequency * np.inner(wave_vector, relative_velocity) * fast_fading)

                # Save the resulting channel coefficients for this ray
                nlos_coefficients[cluster_idx, :, :, :] = (impulse[:, None, None] * channel[None, :, :])

        # In the case of line-of-sight, scale the coefficients and append another set according to equation 7.5-30
        if self.line_of_sight:

            rice_factor_lin = db2lin(rice_factor)
            receiver_position = self.receiver.position
            transmitter_position = self.transmitter.position

            # Raise an exception if the positions are identical
            if np.array_equal(receiver_position, transmitter_position):
                raise RuntimeError("Identical device positions violate the far-field assumption in the line-of-sight"
                                   " case of the 3GPP CDL channel model")

            device_vector = receiver_position - transmitter_position
            los_distance = np.linalg.norm(device_vector, 2)
            rx_wave_vector = device_vector / los_distance * wavelength_factor

            nlos_coefficients *= (1 + rice_factor_lin) ** -.5

            # Equation 7.5-29
            rx_response = self.receiver.antennas.spherical_response(center_frequency, los_aoa, los_zoa)
            tx_response = self.transmitter.antennas.spherical_response(center_frequency, los_aod, los_zod)
            rx_polarization = self.receiver.antennas.polarization(los_aoa, los_zoa)
            tx_polarization = self.transmitter.antennas.polarization(los_aod, los_zod)

            channel = (rx_response[:, None] * rx_polarization) @ (tx_polarization * tx_response[:, None]).T

            impulse = np.exp(2j * pi * center_frequency * np.inner(rx_wave_vector, relative_velocity) * fast_fading)

            los_coefficients = impulse[:, None, None] * channel[None, :, :]

            resampling_matrix = delay_resampling_matrix(sampling_rate, 1, cluster_delays[0],
                                                        num_delay_samples).flatten()
            impulse_response += np.multiply.outer(los_coefficients, resampling_matrix)

        for coefficients, delay in zip(nlos_coefficients, virtual_cluster_delays):
            resampling_matrix = delay_resampling_matrix(sampling_rate, 1, delay, num_delay_samples).flatten()
            impulse_response += np.multiply.outer(coefficients, resampling_matrix)

        return impulse_response

    @property
    def _center_frequency(self) -> float:

        return .5 * (self.transmitter.carrier_frequency + self.receiver.carrier_frequency)


class ClusterDelayLine(ClusterDelayLineBase, Serializable):
    """3GPP Cluster Delay Line Channel Model."""

    yaml_tag = u'ClusterDelayLine'
    """YAML serialization tag."""

    __line_of_sight: bool
    __delay_spread_mean: float
    __delay_spread_std: float
    __aod_spread_mean: float
    __aod_spread_std: float
    __aoa_spread_mean: float
    __aoa_spread_std: float
    __zoa_spread_mean: float
    __zoa_spread_std: float
    __zod_spread_mean: float
    __zod_spread_std: float
    __zod_offset: float
    __rice_factor_mean: float
    __rice_factor_std: float
    __delay_scaling: float
    __cross_polarization_power_mean: float
    __cross_polarization_power_std: float
    __num_clusters: int
    __num_rays: int
    __cluster_delay_spread: float
    __cluster_aod_spread: float
    __cluster_aoa_spread: float
    __cluster_zoa_spread: float
    __cluster_shadowing_std: float

    def __init__(self,
                 line_of_sight: bool = True,
                 delay_spread_mean: float = 7.14,
                 delay_spread_std: float = .38,
                 aod_spread_mean: float = 1.21,
                 aod_spread_std: float = .41,
                 aoa_spread_mean: float = 1.73,
                 aoa_spread_std: float = .28,
                 zoa_spread_mean: float = .73,
                 zoa_spread_std: float = .34,
                 zod_spread_mean: float = .1, 
                 zod_spread_std: float = 0.,
                 zod_offset: float = 0.,
                 rice_factor_mean: float = 9.,
                 rice_factor_std: float = 5.,
                 delay_scaling: float = 1.,
                 cross_polarization_power_mean: float = 9.,
                 cross_polarization_power_std: float = 3.,
                 num_clusters: int = 12,
                 num_rays: int = 20,
                 cluster_delay_spread: float = 5e-9,
                 cluster_aod_spread: float = 5.,
                 cluster_aoa_spread: float = 17.,
                 cluster_zoa_spread: float = 7.,
                 cluster_shadowing_std: float = 3.,
                 **kwargs: Any) -> None:
        """
        Args:

            num_clusters (int, optional):
                Number of generated clusters per channel sample.

            delay_spread (float, optional):
                Root-Mean-Square spread of the cluster delay in seconds.

            delay_scaling (float, optional):
                Delay distribution proportionality factor.

            rice_factor_mean (float, optional):
                Mean of the rice factor K.

            rice_factor_std (float, optional):
                Standard deviation of the rice factor K.

            cluster_shadowing_std (float, optional):
                Cluster shadowing standard deviation in dB.

            line_of_sight (bool, optional):
                Is this model a line-of-sight model?

        """

        # Set initial parameters
        self.line_of_sight = line_of_sight
        self.delay_spread_mean = delay_spread_mean
        self.delay_spread_std = delay_spread_std
        self.aod_spread_mean = aod_spread_mean
        self.aod_spread_std = aod_spread_std
        self.aoa_spread_mean = aoa_spread_mean
        self.aoa_spread_std = aoa_spread_std
        self.zoa_spread_mean = zoa_spread_mean
        self.zoa_spread_std = zoa_spread_std
        self.zod_spread_mean = zod_spread_mean
        self.zod_spread_std = zod_spread_mean
        self.zod_offset = zod_offset
        self.rice_factor_mean = rice_factor_mean
        self.rice_factor_std = rice_factor_std
        self.delay_scaling = delay_scaling
        self.cross_polarization_power_mean = cross_polarization_power_mean
        self.cross_polarization_power_std = cross_polarization_power_std
        self.num_clusters = num_clusters
        self.num_rays = num_rays
        self.cluster_delay_spread = cluster_delay_spread
        self.cluster_aod_spread = cluster_aod_spread
        self.cluster_aoa_spread = cluster_aoa_spread
        self.cluster_zoa_spread = cluster_zoa_spread
        self.cluster_shadowing_std = cluster_shadowing_std

        # Initialize base class
        ClusterDelayLineBase.__init__(self, **kwargs)

    @property
    def line_of_sight(self) -> bool:

        return self.__line_of_sight

    @line_of_sight.setter
    def line_of_sight(self, value: bool) -> None:

        self.__line_of_sight = value

    @property
    def delay_spread_mean(self) -> float:

        return self.__delay_spread_mean

    @delay_spread_mean.setter
    def delay_spread_mean(self, value: float) -> None:

        self.__delay_spread_mean = value
        
    @property
    def delay_spread_std(self) -> float:

        return self.__delay_spread_std

    @delay_spread_std.setter
    def delay_spread_std(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Delay spread standard deviation must be greater or equal to zero")

        self.__delay_spread_std = value

    @property
    def aod_spread_mean(self) -> float:

        return self.__aod_spread_mean

    @aod_spread_mean.setter
    def aod_spread_mean(self, value: float) -> None:

        self.__aod_spread_mean = value

    @property
    def aod_spread_std(self) -> float:

        return self.__aod_spread_std

    @aod_spread_std.setter
    def aod_spread_std(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Angle spread standard deviation must be greater or equal to zero")

        self.__aod_spread_std = value
        
    @property
    def aoa_spread_mean(self) -> float:

        return self.__aoa_spread_mean

    @aoa_spread_mean.setter
    def aoa_spread_mean(self, value: float) -> None:

        self.__aoa_spread_mean = value

    @property
    def aoa_spread_std(self) -> float:

        return self.__aoa_spread_std

    @aoa_spread_std.setter
    def aoa_spread_std(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Angle spread standard deviation must be greater or equal to zero")

        self.__aoa_spread_std = value

    @property
    def zoa_spread_mean(self) -> float:

        return self.__zoa_spread_mean

    @zoa_spread_mean.setter
    def zoa_spread_mean(self, value: float) -> None:

        self.__zoa_spread_mean = value

    @property
    def zoa_spread_std(self) -> float:

        return self.__zoa_spread_std

    @zoa_spread_std.setter
    def zoa_spread_std(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Angle spread standard deviation must be greater or equal to zero")

        self.__zoa_spread_std = value
        
    @property
    def zod_spread_mean(self) -> float:

        return self.__zod_spread_mean

    @zod_spread_mean.setter
    def zod_spread_mean(self, value: float) -> None:

        self.__zod_spread_mean = value

    @property
    def zod_spread_std(self) -> float:

        return self.__zod_spread_std

    @zod_spread_std.setter
    def zod_spread_std(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Zenith spread standard deviation must be greater or equal to zero")

        self.__zod_spread_std = value

    @property
    def zod_offset(self) -> float:
        return self.__zod_offset

    @zod_offset.setter
    def zod_offset(self, value: float) -> None:
        self.__zod_offset = value

    ###############################
    # ToDo: Shadow fading function
    ###############################

    @property
    def rice_factor_mean(self) -> float:

        return self.__rice_factor_mean

    @rice_factor_mean.setter
    def rice_factor_mean(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Rice factor must be greater or equal to zero")

        self.__rice_factor_mean = value

    @property
    def rice_factor_std(self) -> float:

        return self.__rice_factor_std

    @rice_factor_std.setter
    def rice_factor_std(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Rice factor standard deviation must be greater or equal to zero")

        self.__rice_factor_std = value

    @property
    def delay_scaling(self) -> float:

        return self.__delay_scaling

    @delay_scaling.setter
    def delay_scaling(self, value: float) -> None:

        if value < 1.:
            raise ValueError("Delay scaling must be greater or equal to one")

        self.__delay_scaling = value
        
    @property
    def cross_polarization_power_mean(self) -> float:

        return self.__cross_polarization_power_mean

    @cross_polarization_power_mean.setter
    def cross_polarization_power_mean(self, value: float) -> None:

        self.__cross_polarization_power_mean = value

    @property
    def cross_polarization_power_std(self) -> float:

        return self.__cross_polarization_power_std

    @cross_polarization_power_std.setter
    def cross_polarization_power_std(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Cross-polarization power standard deviation must be greater or equal to zero")

        self.__cross_polarization_power_std = value

    @property
    def num_clusters(self) -> int:

        return self.__num_clusters

    @num_clusters.setter
    def num_clusters(self, value: int) -> None:

        if value < 1:
            raise ValueError("Number of clusters must be greater or equal to one")

        self.__num_clusters = value

    @property
    def num_rays(self) -> int:

        return self.__num_rays

    @num_rays.setter
    def num_rays(self, value: int) -> None:

        if value < 1:
            raise ValueError("Number of rays per cluster must be greater or equal to one")

        self.__num_rays = value

    @property
    def cluster_delay_spread(self) -> float:

        return self.__cluster_delay_spread

    @cluster_delay_spread.setter
    def cluster_delay_spread(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Cluster delay spread must be greater or equal to zero")

        self.__cluster_delay_spread = value
        
    @property
    def cluster_aod_spread(self) -> float:

        return self.__cluster_aod_spread

    @cluster_aod_spread.setter
    def cluster_aod_spread(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Cluster angle spread must be greater or equal to zero")

        self.__cluster_aod_spread = value
        
    @property
    def cluster_aoa_spread(self) -> float:

        return self.__cluster_aoa_spread

    @cluster_aoa_spread.setter
    def cluster_aoa_spread(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Cluster angle spread must be greater or equal to zero")

        self.__cluster_aoa_spread = value

    @property
    def cluster_zoa_spread(self) -> float:

        return self.__cluster_zoa_spread

    @cluster_zoa_spread.setter
    def cluster_zoa_spread(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Cluster angle spread must be greater or equal to zero")

        self.__cluster_zoa_spread = value
        
    @property
    def cluster_shadowing_std(self) -> float:

        return self.__cluster_shadowing_std

    @cluster_shadowing_std.setter
    def cluster_shadowing_std(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Cluster shadowing standard deviation must be greater or equal to zero")

        self.__cluster_shadowing_std = value
