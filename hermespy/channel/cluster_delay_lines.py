# -*- coding: utf-8 -*-
"""
=============================
3GPP Cluster Delay Line Model
=============================

Within this module, HermesPy implements the 3GPP standard for cluster delay line models
as defined in :footcite:t:`3GPP:TR38901`.
For a comprehensive description of all the parameters involved, please refer to the standard document.

The abstract base class :class:`.ClusterDelayLineBase` defines all required parameters as abstrat properties
and is required to be implemented by each specific cluster delay line model.
HermesPy features the full customizable model :class:`ClusterDelayLine` as well as
implementations describing standard-compliant benchmark scenarios

=====================================================================   ====================================  ======================================  ========================================
Model                                                                   Line Of Sight                         No Line Of Sight                        Outside To Inside
=====================================================================   ====================================  ======================================  ========================================
:doc:`Indoor Factory <channel.cluster_delay_line_indoor_factory>`       :class:`.IndoorFactoryLineOfSight`    :class:`.IndoorFactoryNoLineOfSight`    *Undefined*
:doc:`Indoor Office <channel.cluster_delay_line_indoor_office>`         :class:`.IndoorOfficeLineOfSight`     :class:`.IndoorOfficeNoLineOfSight`     *Undefined*
:doc:`Rural Macrocells <channel.cluster_delay_line_rural_macrocells>`   :class:`.RuralMacrocellsLineOfSight`  :class:`.RuralMacrocellsNoLineOfSight`  :class:`.RuralMacrocellsOutsideToInside`
:doc:`Street Canyhon <channel.cluster_delay_line_street_canyon>`        :class:`.StreetCanyonLineOfSight`     :class:`.StreetCanyonNoLineOfSight`     :class:`.StreetCanyonOutsideToInside`
:doc:`Urban Macrocells <channel.cluster_delay_line_urban_macrocells>`   :class:`.UrbanMacrocellsLineOfSight`  :class:`.UrbanMacrocellsNoLineOfSight`  :class:`.UrbanMacrocellsOutsideToInside`
=====================================================================   ====================================  ======================================  ========================================

with pre-defined parameters.
In general, the HermesPy cluster delay line implementation mixes deterministic with
statistical information:
:doc:`Devices <simulation.simulated_device>` linked by cluster delay line models are required to specify their assumed
positions and orientations, since the specular line of sight ray components are deterministic.
"""

from __future__ import annotations
from abc import abstractmethod
from enum import Enum
from math import atan, ceil, sin, cos, sqrt
from typing import Any, List, Tuple

import numpy as np
from scipy.constants import pi, speed_of_light

from hermespy.core.factory import Serializable
from hermespy.tools.math import db2lin, rotation_matrix
from hermespy.tools.resampling import delay_resampling_matrix
from .channel import Channel, ChannelRealization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DelayNormalization(Enum):
    """Normalization routine applied to a set of sampled delays.

    Configuration option to :class:.ClusterDelayLineBase models.
    """

    ZERO = 0
    """Normalize the delays, so that the minimal delay is zero"""

    TOF = 1
    """The minimal delay is the time of flight between two devices"""

    NONE = 2
    """No delay normalization is applied.

    Only relevant for debugging purposes.
    """


class ClusterDelayLineBase(Channel):

    delay_normalization: DelayNormalization
    """The delay normalization routine applied during channel sampling."""

    # Cluster scaling factors for the angle of arrival
    __azimuth_scaling_factors = np.array([[4, 0.779], [5, 0.86], [8, 1.018], [10, 1.090], [11, 1.123], [12, 1.146], [14, 1.19], [15, 1.211], [16, 1.226], [19, 1.273], [20, 1.289]], dtype=float)

    __zenith_scaling_factors = np.array([[8, 0.889], [10, 0.957], [11, 1.031], [12, 1.104], [15, 1.108], [19, 1.184], [20, 1.178]], dtype=float)

    # Ray offset angles
    __ray_offset_angles = np.array([0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129, 0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551])

    # Sub-cluster partitions for the three strongest clusters
    __subcluster_indices: List[List[int]] = [[0, 1, 2, 3, 4, 5, 6, 7, 18, 19], [8, 9, 10, 11, 16, 17], [12, 13, 14, 15]]

    def __init__(self, delay_normalization: DelayNormalization = DelayNormalization.ZERO, **kwargs) -> None:
        """
        Args:

            delay_normalization (DelayNormalization, optional):

                The delay normalization routine applied during channel sampling.
        """

        self.delay_normalization = delay_normalization

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

    def _cluster_delays(self, delay_spread: float, rice_factor: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a single sample set of normalized cluster delays.

        A single cluster delay is referred to as :math:`\\tau_n` within the the standard.

        Args:

            delay_spread (float):
                Delay spread in seconds.

            rice_factor (float):
                Rice factor K in dB.

        Returns:

            Tuple of
                - DRaw delay samples
                - True delays
        """

        # Generate delays according to the configured spread and scales
        raw_delays = -self.delay_scaling * delay_spread * np.log(self._rng.uniform(size=self.num_clusters))

        # Sort the delays in ascending order
        raw_delays.sort()

        # Normalize delays if the respective flag is enabled
        if self.delay_normalization == DelayNormalization.ZERO or self.delay_normalization == DelayNormalization.TOF:

            raw_delays -= raw_delays[0]

        # Scale delays, if required by the configuration
        scaled_delays = raw_delays.copy()

        # In case of line of sight, scale the delays by the appropriate K-factor
        if self.line_of_sight:

            rice_scale = 0.775 - 0.0433 * rice_factor + 2e-4 * rice_factor**2 + 17e-6 * rice_factor**3
            scaled_delays /= rice_scale

        # Account for the time of flight over the line of sight, if required
        if self.delay_normalization == DelayNormalization.TOF:

            time_of_flight = np.linalg.norm(self.transmitter.position - self.receiver.position, 2) / speed_of_light
            scaled_delays += time_of_flight

        # Return the raw and scaled delays, since they are both required for further processing
        return raw_delays, scaled_delays

    def _cluster_powers(self, delay_spread: float, delays: np.ndarray, rice_factor: float) -> np.ndarray:
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

        shadowing = 10 ** (-0.1 * self._rng.normal(scale=self.cluster_shadowing_std, size=delays.shape))
        powers = np.exp(-delays * (self.delay_scaling - 1) / (self.delay_scaling * delay_spread)) * shadowing

        # In case of line of sight, add a specular component to the cluster delays
        if self.line_of_sight:

            linear_rice_factor = db2lin(rice_factor)
            powers /= (1 + linear_rice_factor) * np.sum(powers.flat)
            powers[0] += linear_rice_factor / (1 + linear_rice_factor)

        else:
            powers /= np.sum(powers.flat)

        return powers

    def _ray_azimuth_angles(self, cluster_powers: np.ndarray, rice_factor: float, los_azimuth: float) -> np.ndarray:
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
            angle_scale *= 1.1035 - 0.028 * rice_factor - 2e-3 * rice_factor**2 + 1e-4 * rice_factor**3

        # Draw azimuth angle spread from the distribution
        spread = 10 ** self._rng.normal(self.aoa_spread_mean, self.aoa_spread_std, size=size)

        angles = 2 * (spread / 1.4) * np.sqrt(-np.log(cluster_powers / cluster_powers.max())) / angle_scale

        # Assign positive / negative integers and add some noise
        angle_variation = self._rng.normal(0.0, (spread / 7) ** 2, size=size)
        angle_spread_sign = self._rng.choice([-1.0, 1.0], size=size)
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

    def _ray_zoa(self, cluster_powers: np.ndarray, rice_factor: float, los_zenith: float) -> np.ndarray:
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
            zenith_scale *= 1.3086 + 0.0339 * rice_factor - 0.0077 * rice_factor**2 + 2e-4 * rice_factor**3

        # Draw zenith angle spread from the distribution
        zenith_spread = 10 ** self._rng.normal(self.zoa_spread_mean, self.zoa_spread_std, size=size)

        # Generate angle starting point
        cluster_zenith = -zenith_spread * np.log(cluster_powers / cluster_powers.max()) / zenith_scale

        cluster_variation = self._rng.normal(0.0, (zenith_spread / 7) ** 2, size=size)
        cluster_sign = self._rng.choice([-1.0, 1.0], size=size)

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

    def _ray_zod(self, cluster_powers: np.ndarray, rice_factor: float, los_zenith: float) -> np.ndarray:
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
            zenith_scale *= 1.3086 + 0.0339 * rice_factor - 0.0077 * rice_factor**2 + 2e-4 * rice_factor**3

        # Draw zenith angle spread from the distribution
        zenith_spread = 10 ** self._rng.normal(self.zoa_spread_mean, self.zoa_spread_std, size=size)

        # Generate angle starting point
        cluster_zenith = -zenith_spread * np.log(cluster_powers / cluster_powers.max()) / zenith_scale

        cluster_variation = self._rng.normal(0.0, (zenith_spread / 7) ** 2, size=size)
        cluster_sign = self._rng.choice([-1.0, 1.0], size=size)

        # ToDo: Treat the BST-UT case!!!! (los_zenith = 90°)
        # Equation 7.5-19
        cluster_zenith: np.ndarray = cluster_sign * cluster_zenith + cluster_variation + self.zod_offset

        if self.line_of_sight:
            cluster_zenith += los_zenith - cluster_zenith[0]

        else:
            cluster_zenith += los_zenith

        # Spread the angles
        # Equation 7.5 -20
        ray_offsets = 3 / 8 * 10**self.zoa_spread_mean * self.__ray_offset_angles
        ray_zenith = np.tile(cluster_zenith[:, None], len(ray_offsets)) + ray_offsets

        return ray_zenith

    def realize(self,
                num_samples: int,
                sampling_rate: float) -> ChannelRealization:

        center_frequency = self.transmitter.carrier_frequency

        delay_spread = 10 ** self._rng.normal(self.delay_spread_mean, self.delay_spread_std)
        rice_factor = self._rng.normal(loc=self.rice_factor_mean, scale=self.rice_factor_std)

        # Query device positions and orientations
        tx_position = self.transmitter.position
        rx_position = self.receiver.position
        tx_orientation = self.transmitter.orientation  # Orientation in RPY
        rx_orientation = self.receiver.orientation  # Orientation in RPY

        # Positions may not be unspecified
        if tx_position is None or rx_position is None:
            raise ValueError("Cluster delay line models require specified transmitter and receiver positions")

        # Compute the respective angles of arrival and departure
        tx_los_vector = rotation_matrix(-tx_orientation) @ (rx_position - tx_position)
        rx_los_vector = rotation_matrix(-rx_orientation) @ (tx_position - rx_position)

        los_aoa = atan(rx_los_vector[1] / rx_los_vector[0]) if rx_los_vector[1] != 0.0 and rx_los_vector[0] != 0.0 else 0.0
        los_aod = atan(tx_los_vector[1] / tx_los_vector[0]) if tx_los_vector[1] != 0.0 and tx_los_vector[0] != 0.0 else 0
        los_zoa = atan(sqrt(rx_los_vector[0] ** 2 + rx_los_vector[1] ** 2) / rx_los_vector[2]) if rx_los_vector[2] != 0.0 else 0.5 * pi
        los_zod = atan(sqrt(tx_los_vector[0] ** 2 + tx_los_vector[1] ** 2) / tx_los_vector[2]) if tx_los_vector[2] != 0.0 else 0.5 * pi

        num_clusters = self.num_clusters
        num_rays = 20

        raw_cluster_delays, cluster_delays = self._cluster_delays(delay_spread, rice_factor)
        cluster_powers = self._cluster_powers(delay_spread, raw_cluster_delays, rice_factor)

        ray_aod = pi / 180 * self._ray_azimuth_angles(cluster_powers, rice_factor, 180 * los_aod / pi)
        ray_aoa = pi / 180 * self._ray_azimuth_angles(cluster_powers, rice_factor, 180 * los_aoa / pi)
        # ToDo: Zenith departure modeling
        ray_zod = pi / 180 * self._ray_zod(cluster_powers, rice_factor, 180 * los_zod / pi)
        ray_zoa = pi / 180 * self._ray_zoa(cluster_powers, rice_factor, 180 * los_zoa / pi)

        # Couple cluster angles randomly (step 8)
        # This is equivalent to shuffeling the angles within each cluster set
        for ray_angles in (ray_aod, ray_aoa, ray_zod, ray_zoa):
            [self._rng.shuffle(a) for a in ray_angles]

        # Generate cross-polarization power ratios (step 9)
        xpr = 10 ** (0.1 * self._rng.normal(self.cross_polarization_power_mean, self.cross_polarization_power_std, size=(num_clusters, num_rays)))

        # Draw initial random phases (step 10)
        jones_matrix = np.exp(2j * pi * self._rng.uniform(size=(2, 2, num_clusters, num_rays)))
        jones_matrix[0, 1, ::] *= xpr**-0.5
        jones_matrix[1, 0, ::] *= xpr**-0.5

        # Initialize channel matrices
        num_delay_samples = 1 + ceil(cluster_delays.max() * sampling_rate)
        impulse_response = np.zeros((num_samples, self.receiver.antennas.num_antennas, self.transmitter.antennas.num_antennas, num_delay_samples), dtype=complex)

        # Compute the number of clusters, considering the first two clusters get split into 3 partitions
        num_split_clusters = min(2, num_clusters)
        virtual_num_clusters = 3 * num_split_clusters + max(0, num_clusters - 2)

        # Prepare the channel coefficient storage
        nlos_coefficients = np.zeros((virtual_num_clusters, num_samples, self.receiver.antennas.num_antennas, self.transmitter.antennas.num_antennas), dtype=complex)

        # Prepare the cluster delays, equation 7.5-26
        subcluster_delays = np.repeat(cluster_delays[:num_split_clusters, None], 3, axis=1) + self.cluster_delay_spread * np.array([1.0, 1.28, 2.56])
        virtual_cluster_delays = np.concatenate((subcluster_delays.flatten(), cluster_delays[num_split_clusters:]))

        # Wavelength factor
        wavelength_factor = self.transmitter.carrier_frequency / speed_of_light
        relative_velocity = self.receiver.velocity - self.transmitter.velocity
        fast_fading = wavelength_factor * np.arange(num_samples) / sampling_rate

        for subcluster_idx in range(0, virtual_num_clusters):

            cluster_idx = int(subcluster_idx / 3) if subcluster_idx < 6 else subcluster_idx - 4
            ray_indices = self.__subcluster_indices[cluster_idx] if cluster_idx < num_split_clusters else range(num_rays)

            for aoa, zoa, aod, zod, jones in zip(ray_aoa[cluster_idx, ray_indices], ray_zoa[cluster_idx, ray_indices], ray_aod[cluster_idx, ray_indices], ray_zod[cluster_idx, ray_indices], jones_matrix[:, :, cluster_idx, ray_indices].transpose(2, 0, 1)):

                # Equation 7.5-23
                rx_response = self.receiver.antennas.spherical_response(center_frequency, aoa, zoa)

                # Equation 7.5-24
                tx_response = self.transmitter.antennas.spherical_response(center_frequency, aod, zod).conj()

                # Equation 7.5-28
                rx_polarization = self.receiver.antennas.polarization(aoa, zoa)
                tx_polarization = self.transmitter.antennas.polarization(aod, zod)

                channel = (rx_response[:, None] * rx_polarization) @ jones @ (tx_polarization * tx_response[:, None]).T * sqrt(cluster_powers[cluster_idx] / num_clusters)

                wave_vector = np.array([cos(aoa) * sin(zoa), sin(aoa) * sin(zoa), cos(zoa)], dtype=float)
                impulse = np.exp(np.inner(wave_vector, relative_velocity) * fast_fading * 2j * pi)

                # Save the resulting channel coefficients for this ray
                nlos_coefficients[subcluster_idx, :, :, :] = impulse[:, None, None] * channel[None, :, :]

        # In the case of line-of-sight, scale the coefficients and append another set according to equation 7.5-30
        if self.line_of_sight:

            rice_factor_lin = db2lin(rice_factor)
            receiver_position = self.receiver.position
            transmitter_position = self.transmitter.position

            # Raise an exception if the positions are identical
            if np.array_equal(receiver_position, transmitter_position):
                raise RuntimeError("Identical device positions violate the far-field assumption in the line-of-sight" " case of the 3GPP CDL channel model")

            device_vector = receiver_position - transmitter_position
            los_distance = np.linalg.norm(device_vector, 2)
            rx_wave_vector = device_vector / los_distance

            # First summand scaling of equation 7.5-30
            nlos_coefficients *= (1 + rice_factor_lin) ** -0.5

            # Equation 7.5-29
            rx_response = self.receiver.antennas.spherical_response(center_frequency, los_aoa, los_zoa)
            tx_response = self.transmitter.antennas.spherical_response(center_frequency, los_aod, los_zod).conj()
            rx_polarization = self.receiver.antennas.polarization(los_aoa, los_zoa)
            tx_polarization = self.transmitter.antennas.polarization(los_aod, los_zod)

            channel = (rx_response[:, None] * rx_polarization) @ (tx_polarization * tx_response[:, None]).T
            impulse = np.exp(-2j * pi * los_distance * wavelength_factor) * np.exp(np.inner(rx_wave_vector, relative_velocity) * fast_fading * 2j * pi)

            los_coefficients = impulse[:, None, None] * channel[None, :, :]

            # Second summand of equation 7.5-30
            resampling_matrix = delay_resampling_matrix(sampling_rate, 1, cluster_delays[0], num_delay_samples).flatten()
            impulse_response += (rice_factor_lin / 1 + rice_factor_lin) ** 0.5 * np.multiply.outer(los_coefficients, resampling_matrix)

        # Finally, generate the impulse response for all non-line of sight components
        for coefficients, delay in zip(nlos_coefficients, virtual_cluster_delays):

            resampling_matrix = delay_resampling_matrix(sampling_rate, 1, delay, num_delay_samples).flatten()
            impulse_response += np.multiply.outer(coefficients, resampling_matrix)

        return ChannelRealization(self, impulse_response)

    @property
    def _center_frequency(self) -> float:

        return 0.5 * (self.transmitter.carrier_frequency + self.receiver.carrier_frequency)


class ClusterDelayLine(ClusterDelayLineBase, Serializable):
    """3GPP Cluster Delay Line Channel Model."""

    yaml_tag = "ClusterDelayLine"
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

    def __init__(
        self,
        line_of_sight: bool = True,
        delay_spread_mean: float = 7.14,
        delay_spread_std: float = 0.38,
        aod_spread_mean: float = 1.21,
        aod_spread_std: float = 0.41,
        aoa_spread_mean: float = 1.73,
        aoa_spread_std: float = 0.28,
        zoa_spread_mean: float = 0.73,
        zoa_spread_std: float = 0.34,
        zod_spread_mean: float = 0.1,
        zod_spread_std: float = 0.0,
        zod_offset: float = 0.0,
        rice_factor_mean: float = 9.0,
        rice_factor_std: float = 5.0,
        delay_scaling: float = 1.0,
        cross_polarization_power_mean: float = 9.0,
        cross_polarization_power_std: float = 3.0,
        num_clusters: int = 12,
        num_rays: int = 20,
        cluster_delay_spread: float = 5e-9,
        cluster_aod_spread: float = 5.0,
        cluster_aoa_spread: float = 17.0,
        cluster_zoa_spread: float = 7.0,
        cluster_shadowing_std: float = 3.0,
        **kwargs: Any,
    ) -> None:
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
        self.zod_spread_std = zod_spread_std
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

        if value < 0.0:
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

        if value < 0.0:
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

        if value < 0.0:
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

        if value < 0.0:
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

        if value < 0.0:
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

        if value < 0.0:
            raise ValueError("Rice factor must be greater or equal to zero")

        self.__rice_factor_mean = value

    @property
    def rice_factor_std(self) -> float:

        return self.__rice_factor_std

    @rice_factor_std.setter
    def rice_factor_std(self, value: float) -> None:

        if value < 0.0:
            raise ValueError("Rice factor standard deviation must be greater or equal to zero")

        self.__rice_factor_std = value

    @property
    def delay_scaling(self) -> float:

        return self.__delay_scaling

    @delay_scaling.setter
    def delay_scaling(self, value: float) -> None:

        if value < 1.0:
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

        if value < 0.0:
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

        if value < 0.0:
            raise ValueError("Cluster delay spread must be greater or equal to zero")

        self.__cluster_delay_spread = value

    @property
    def cluster_aod_spread(self) -> float:

        return self.__cluster_aod_spread

    @cluster_aod_spread.setter
    def cluster_aod_spread(self, value: float) -> None:

        if value < 0.0:
            raise ValueError("Cluster angle spread must be greater or equal to zero")

        self.__cluster_aod_spread = value

    @property
    def cluster_aoa_spread(self) -> float:

        return self.__cluster_aoa_spread

    @cluster_aoa_spread.setter
    def cluster_aoa_spread(self, value: float) -> None:

        if value < 0.0:
            raise ValueError("Cluster angle spread must be greater or equal to zero")

        self.__cluster_aoa_spread = value

    @property
    def cluster_zoa_spread(self) -> float:

        return self.__cluster_zoa_spread

    @cluster_zoa_spread.setter
    def cluster_zoa_spread(self, value: float) -> None:

        if value < 0.0:
            raise ValueError("Cluster angle spread must be greater or equal to zero")

        self.__cluster_zoa_spread = value

    @property
    def cluster_shadowing_std(self) -> float:

        return self.__cluster_shadowing_std

    @cluster_shadowing_std.setter
    def cluster_shadowing_std(self, value: float) -> None:

        if value < 0.0:
            raise ValueError("Cluster shadowing standard deviation must be greater or equal to zero")

        self.__cluster_shadowing_std = value
