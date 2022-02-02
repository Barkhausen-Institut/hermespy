# -*- coding: utf-8 -*-
"""
=============================
3GPP Cluster Delay Line Model
=============================
"""

from __future__ import annotations
from abc import abstractmethod
from itertools import product
from math import ceil, sin, cos, sqrt
from typing import Any, List

import numpy as np
from scipy.constants import pi, speed_of_light

from ..core.factory import Serializable
from ..tools.math import db2lin
from ..helpers.resampling import delay_resampling_matrix
from .channel import Channel

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ClusterDelayLine(Channel, Serializable):

    yaml_tag = u'ClusterDelayLine'
    """YAML serialization tag."""

    line_of_sight: bool
    """Is this model a line of sight model?"""

    __num_clusters: int             # Number of generated clusters per channel sample
    __delay_spread: float           # Root-Mean-Square spread of the cluster delay in seconds
    __delay_scaling: float          # Delay distribution proportionality factor
    __rice_factor_mean: float       # Mean of the rice factor K
    __rice_factor_std: float        # Standard deviation of the rice factor K
    __cluster_shadowing_std: float  # Cluster shadowing standard deviation in dB

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

    def __init__(self,
                 num_clusters: int = 10,
                 delay_spread: float = 10e-9,
                 delay_scaling: float = 1.,
                 rice_factor_mean: float = 7.,
                 rice_factor_std: float = 4.,
                 cluster_shadowing_std: float = 3.,
                 line_of_sight: bool = False,
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

        self.line_of_sight = line_of_sight
        self.num_clusters = num_clusters
        self.delay_spread = delay_spread
        self.delay_scaling = delay_scaling
        self.rice_factor_mean = rice_factor_mean
        self.rice_factor_std = rice_factor_std
        self.cluster_shadowing_std = cluster_shadowing_std

        # Initialize base class
        Channel.__init__(self, **kwargs)

    @property
    def num_clusters(self) -> int:
        """Number of clusters.

        Returns:
            int: Number of generated clusters per channel sample.

        Raises:
            ValueError: If the number of clusters is smaller than one.
        """

        return self.__num_clusters

    @num_clusters.setter
    def num_clusters(self, value: int) -> None:

        if value < 1:
            raise ValueError("Number of clusters must be greater or equal to one")

        self.__num_clusters = value

    @property
    def delay_spread(self) -> float:
        """Root-Mean-Square spread of the cluster delay.

        Referred to as :math:`\\mathrm{DS}` within the the standard.

        Returns:
            float: Delay spread in seconds.

        Raises:
            ValueError: If the delay spread is smaller than zero.
        """

        return self.__delay_spread

    @property
    def intra_cluster_delay_spread(self) -> float:
        """Delay spread within an individual cluster.

        Referred to as :math:`c_{DS}` within the standard.

        Returns:
            float: Delay spread in seconds.
        """

        return 3.91e-9

    @delay_spread.setter
    def delay_spread(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Delay spread must be greater or equal to zero")

        self.__delay_spread = value

    @property
    def delay_scaling(self) -> float:
        """Delay distribution proportionality factor.

        Referred to as :math:`r_{\\tau}` within the standard.

        Returns:
            float: Scaling factor.

        Raises:
            ValueError:
                If scaling factor is smaller than one.
        """

        return self.__delay_scaling

    @delay_scaling.setter
    def delay_scaling(self, value: float) -> None:

        if value < 1.:
            raise ValueError("Delay scaling must be greater or equal to one")

        self.__delay_scaling = value
        
    @property
    def rice_factor_mean(self) -> float:
        """Mean of the rice factor distribution.

        Referred to as :math:`\\mu_K` within the the standard.

        Returns:
            float: Rice factor mean

        Raises:
            ValueError: If the mean is smaller than zero.
        """

        return self.__rice_factor_mean

    @rice_factor_mean.setter
    def rice_factor_mean(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Rice factor mean must be greater or equal to zero")

        self.__rice_factor_mean = value
        
    @property
    def rice_factor_std(self) -> float:
        """Standard deviation of the rice factor distribution.

        Referred to as :math:`\\sigma_K` within the the standard.

        Returns:
            float: Rice factor standard deviation.

        Raises:
            ValueError: If the deviation is smaller than zero.
        """

        return self.__rice_factor_std

    @rice_factor_std.setter
    def rice_factor_std(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Rice factor standard deviation must be greater or equal to zero")

        self.__rice_factor_std = value
        
    @property
    def cluster_shadowing_std(self) -> float:
        """Standard deviation of the cluster shadowing.

        Referred to as ??? within the the standard.

        Returns:
            float: Cluster shadowing standard deviation.

        Raises:
            ValueError: If the deviation is smaller than zero.
        """

        return self.__cluster_shadowing_std

    @cluster_shadowing_std.setter
    def cluster_shadowing_std(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Cluster shadowing standard deviation must be greater or equal to zero")

        self.__cluster_shadowing_std = value

    @property
    @abstractmethod
    def aod_spread_mean(self) -> float:
        """Azimuth Angle-of-Departure spread mean.

        Returns:
            float: Mean spread in degrees.
        """
        ...

    @property
    @abstractmethod
    def aod_spread_std(self) -> float:
        """Azimuth Angle-of-Departure spread standard deviation.

        Returns:
            float: Spread standard deviation in degrees.
        """
        ...

    @property
    @abstractmethod
    def aoa_spread_mean(self) -> float:
        """Angle-of-Arrival spread mean.

        Returns:
            float: Mean spread in degrees.
        """
        ...

    @property
    @abstractmethod
    def aoa_spread_std(self) -> float:
        """Angle-of-Arrival spread standard deviation.

        Returns:
            float: Spread standard deviation in degrees.
        """
        ...

    @property
    @abstractmethod
    def zoa_spread_mean(self) -> float:
        """Zenith Angle-of-Arrival spread mean.

        Returns:
            float: Mean spread in degrees.
        """
        ...

    @property
    @abstractmethod
    def zoa_spread_std(self) -> float:
        """Zenith Angle-of-Arrival spread standard deviation.

        Returns:
            float: Spread standard deviation in degrees.
        """
        ...

    @property
    @abstractmethod
    def cluster_azimuth_spread_departure(self) -> float:
        """Cluster azimuth ray spread during departure.

        Returns:
            float: Spread in deg.
        """
        ...

    @property
    @abstractmethod
    def cluster_azimuth_spread_arrival(self) -> float:
        """Cluster azimuth ray spread during arrival.

        Returns:
            float: Spread in deg.
        """
        ...

    @property
    @abstractmethod
    def cluster_zenith_spread_arrival(self) -> float:
        """Cluster zenith ray spread during arrival.

        Returns:
            float: Spread in deg.
        """
        ...

    @property
    @abstractmethod
    def cross_polarization_power_mean(self) -> float:
        """Mean of the cross polarization power.

        Returns:
            float: Power mean.
        """
        ...

    @property
    @abstractmethod
    def cross_polarization_power_std(self) -> float:
        """Standard deviation of the cross polarization power.

        Returns:
            float: Power standard deviation.
        """
        ...

    def _cluster_delays(self,
                        rice_factor: float) -> np.ndarray:
        """Compute a single sample set of normalized cluster delays.

        A single cluster delay is referred to as :math:`\\tau_n` within the the standard.

        Args:

            rice_factor (float):
                Rice factor K in dB.

        Returns:

            np.ndarray:
                Vector of cluster delays.
        """

        delays = - self.delay_scaling * self.delay_spread * np.log(self._rng.uniform(size=self.num_clusters))

        delays -= delays.min()
        delays.sort()

        # In case of line of sight, scale the delays by the appropriate K-factor
        if self.line_of_sight:

            rice_scale = .775 - .0433 * rice_factor + 2e-4 * rice_factor ** 2 + 17e-6 * rice_factor ** 3
            delays /= rice_scale

        return delays

    def _cluster_powers(self,
                        delays: np.ndarray,
                        rice_factor: float) -> np.ndarray:
        """Compute a single sample set of normalized cluster power factors from delays.

        A single cluster power factor is referred to as :math:`P_n` within the the standard.

        Args:

            delays (np.ndarray):
                Vector of cluster delays.

            rice_factor (float):
                Rice factor K in dB.

        Returns:

            np.ndarray:
                Vector of cluster power scales.
        """

        shadowing = 10 ** (-.1 * self._rng.normal(scale=self.cluster_shadowing_std, size=delays.shape))
        powers = np.exp(-delays * (self.delay_scaling - 1) / (self.delay_scaling * self.delay_spread)) * shadowing

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
        spread = self._rng.normal(self.aoa_spread_mean, self.aoa_spread_std, size=size)

        angles = 2 * (spread / 1.4) * np.sqrt(-np.log(cluster_powers / cluster_powers.max())) / angle_scale

        # Assign positive / negative integers and add some noise
        angle_variation = self._rng.normal(0., (spread / 7) ** 2, size=size)
        angle_spread_sign = self._rng.choice([-1., 1.], size=size)
        angles = angle_spread_sign * angles + angle_variation

        # Add the actual line of sight term
        if self.line_of_sight:

            # The first angle within the list is exactly the line of sight component
            angles += los_azimuth - angles[0]

        else:

            angles += los_azimuth

        # Spread the angles
        ray_offsets = self.cluster_azimuth_spread_arrival * self.__ray_offset_angles
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
        zenith_spread = self._rng.normal(self.zoa_spread_mean, self.zoa_spread_std, size=size)

        # Generate angle starting point
        cluster_zenith = -zenith_spread * np.log(cluster_powers / cluster_powers.max()) / zenith_scale

        cluster_variation = self._rng.normal(0., (zenith_spread / 7) ** 2, size=size)
        cluster_sign = self._rng.choice([-1., 1.], size=size)

        # ToDo: Treat the BST-UT case!!!! (los_zenith = 90°)
        cluster_zenith = cluster_sign * cluster_zenith + cluster_variation

        if self.line_of_sight:
            cluster_zenith += los_zenith - cluster_zenith[0]

        else:
            cluster_zenith += los_zenith

        # Spread the angles
        ray_offsets = self.cluster_zenith_spread_arrival * self.__ray_offset_angles
        ray_zenith = np.tile(cluster_zenith[:, None], len(ray_offsets)) + ray_offsets

        return ray_zenith

    def impulse_response(self,
                         num_samples: int,
                         sampling_rate: float) -> np.ndarray:

        rice_factor = self._rng.normal(loc=self.rice_factor_mean, scale=self.rice_factor_std)
        los_aoa = 0.
        los_aod = 0.
        los_zoa = 0.
        los_zod = 0.

        num_clusters = self.num_clusters
        num_rays = 20

        cluster_delays = self._cluster_delays(rice_factor)
        cluster_powers = self._cluster_powers(cluster_delays, rice_factor)

        ray_aod = self._ray_azimuth_angles(cluster_powers, rice_factor, los_aod)
        ray_aoa = self._ray_azimuth_angles(cluster_powers, rice_factor, los_aoa)
        ray_zod = self._ray_zoa(cluster_powers, rice_factor, los_zod)   # ToDo: Zenith departure modeling
        ray_zoa = self._ray_zoa(cluster_powers, rice_factor, los_zoa)

        # Cuple angles ToDo
        #azimuth_ray_couples =

        # Generate cross-polarization power ratios (step 9)
        xpr = 10 ** (.1 * self._rng.normal(self.cross_polarization_power_mean,
                                           self.cross_polarization_power_std,
                                           size=(num_clusters, num_rays)))

        # Draw initial random phases (step 10)
        phases = np.exp(2j * pi * self._rng.uniform(size=(2, 2, num_clusters, num_rays)))
        phases[0, 1, ::] *= xpr ** -.5
        phases[1, 0, ::] *= xpr ** -.5

        # Initialize channel matrices
        num_delay_samples = 1 + ceil(cluster_delays.max() * sampling_rate)
        impulse_response = np.ndarray((num_samples, self.receiver.num_antennas,
                                       self.transmitter.num_antennas, num_delay_samples), dtype=complex)

        # Compute the number of clusters, considering the first two clusters get split into 3 partitions
        num_split_clusters = min(2, num_clusters)
        virtual_num_clusters = 3 * num_split_clusters + max(0, num_clusters - 2)

        # Prepare the channel coefficient storage
        nlos_coefficients = np.zeros((virtual_num_clusters, num_samples, self.receiver.num_antennas,
                                     self.transmitter.num_antennas), dtype=complex)

        # Prepare the cluster delays, equation 7.5-26
        subcluster_delays = (np.repeat(cluster_delays[:num_split_clusters, None], 3, axis=1) +
                             self.intra_cluster_delay_spread * np.array([1., 1.28, 2.56]))
        virtual_cluster_delays = np.concatenate((subcluster_delays.flatten(), cluster_delays[num_split_clusters:]))

        # Weak cluster coefficients
        rx_positions = self.receiver.antenna_positions
        tx_positions = self.transmitter.antenna_positions

        # Wavelength factor
        wavelength_factor = 2j * pi * self.transmitter.carrier_frequency / speed_of_light
        fast_fading = wavelength_factor * np.outer(self.receiver.velocity, np.arange(num_samples) / sampling_rate)

        for cluster_idx in range(0, num_split_clusters):

            for subcluster_idx, ray_indices in enumerate(self.__subcluster_indices):
                for ray_idx in ray_indices:

                    # Equation 7.5-23
                    rx_wave_vector = np.array([sin(ray_zoa[cluster_idx, ray_idx]) * cos(ray_aoa[cluster_idx, ray_idx]),
                                               sin(ray_zoa[cluster_idx, ray_idx]) * sin(ray_aoa[cluster_idx, ray_idx]),
                                               cos(ray_zoa[cluster_idx, ray_idx])]) * wavelength_factor

                    # Equation 7.5-24
                    tx_wave_vector = np.array([sin(ray_zod[cluster_idx, ray_idx]) * cos(ray_aod[cluster_idx, ray_idx]),
                                               sin(ray_zod[cluster_idx, ray_idx]) * sin(ray_aod[cluster_idx, ray_idx]),
                                               cos(ray_zod[cluster_idx, ray_idx])]) * wavelength_factor

                    # Equation 7.5-28
                    for (rx_idx, rx_pos), (tx_idx, tx_pos) in product(enumerate(rx_positions),
                                                                      enumerate(tx_positions)):

                        ray_coefficients = (np.exp(rx_wave_vector @ rx_pos) * np.exp(tx_wave_vector @ tx_pos) *
                                            complex(np.ones((2, 1)).T @ phases[:, :, cluster_idx, ray_idx] @ np.ones((2, 1))) *
                                            np.exp(rx_wave_vector @ fast_fading).T) * sqrt(cluster_powers[cluster_idx] / num_clusters)

                        nlos_coefficients[3 * cluster_idx + subcluster_idx, :, rx_idx, tx_idx] = ray_coefficients

        for cluster_idx, ray_idx in product(range(num_clusters - 2), range(num_rays)):

            # Equation 7.5-23
            rx_wave_vector = np.array([sin(ray_zoa[cluster_idx, ray_idx]) * cos(ray_aoa[cluster_idx, ray_idx]),
                                       sin(ray_zoa[cluster_idx, ray_idx]) * sin(ray_aoa[cluster_idx, ray_idx]),
                                       cos(ray_zoa[cluster_idx, ray_idx])]) * wavelength_factor

            # Equation 7.5-24
            tx_wave_vector = np.array([sin(ray_zod[cluster_idx, ray_idx]) * cos(ray_aod[cluster_idx, ray_idx]),
                                       sin(ray_zod[cluster_idx, ray_idx]) * sin(ray_aod[cluster_idx, ray_idx]),
                                       cos(ray_zod[cluster_idx, ray_idx])]) * wavelength_factor

            # Equation 7.5-22
            for (rx_idx, rx_pos), (tx_idx, tx_pos) in product(enumerate(rx_positions),
                                                              enumerate(tx_positions)):

                ray_coefficients = (np.exp(rx_wave_vector @ rx_pos) * np.exp(tx_wave_vector @ tx_pos) *
                                    complex(np.ones((2, 1)).T @ phases[:, :, cluster_idx, ray_idx] @ np.ones((2, 1))) *
                                    np.exp(rx_wave_vector @ fast_fading).T)
                nlos_coefficients[6 + cluster_idx, :, rx_idx, tx_idx] += ray_coefficients * sqrt(cluster_powers[cluster_idx] / num_rays)

        for coefficients, delay in zip(nlos_coefficients, virtual_cluster_delays):

            resampling_matrix = delay_resampling_matrix(sampling_rate, 1, delay, num_delay_samples).flatten()
            impulse_response += np.multiply.outer(coefficients, resampling_matrix)

        return impulse_response
