# -*- coding: utf-8 -*-
"""
=============================
3GPP Cluster Delay Line Model
=============================
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Tuple

import numpy as np

from ..core.factory import Serializable
from ..tools.math import db2lin
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

    line_of_sight: bool
    """Is this model a line of sight model?"""

    __num_clusters: int             # Number of generated clusters per channel sample
    __delay_spread: float           # Root-Mean-Square spread of the cluster delay in seconds
    __delay_scaling: float          # Delay distribution proportionality factor
    __rice_factor_mean: float       # Mean of the rice factor K
    __rice_factor_std: float        # Standard deviation of the rice factor K
    __cluster_shadowing_std: float  # Cluster shadowing standard deviation in dB

    # Cluster scaling factors for the angle of arrival
    __cluster_scaling_factors = np.array([[4, .779],
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

    def __init__(self,
                 num_clusters: int = 10,
                 delay_spread: float = 10e-9,
                 delay_scaling: float = 1.,
                 rice_factor_mean: float = 7.,
                 rice_factor_std: float = 4.,
                 cluster_shadowing_std: float = 3.,
                 line_of_sight: bool = False) -> None:
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
        Channel.__init__(self)

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
    def aod_spread_mean(self) -> float:
        """Angle-of-Departure spread mean.

        Returns:
            float: Mean spread in degrees.
        """
        ...

    @property
    @abstractmethod
    def aod_spread_std(self) -> float:
        """Angle-of-Departure spread standard deviation.

        Returns:
            float: Spread standard deviation in degrees.
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
                        rice_factor: float) -> None:
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

    def _cluster_azimuth_angles(self,
                                cluster_powers: np.ndarray,
                                rice_factor: float) -> np.ndarray:

        # Determine the closest scaling factor
        scale_index = np.argmin(np.abs(self.__cluster_scaling_factors[:, 0] - len(cluster_powers)))
        angle_scale = self.__cluster_scaling_factors[scale_index, 1]

        # Scale the scale (hehe) in the line of sight case
        if self.line_of_sight:
            angle_scale *= 1.1035 - .028 * rice_factor - 2e-3 * rice_factor ** 2 + 1e-4 * rice_factor ** 3

        # Draw angle spread from the distribution
        angle_spread = self._rng.normal(self.aoa_spread_mean, self.aoa_spread_std, size=cluster_powers.shape)

        angles = 2 * (angle_spread / 1.4) * np.sqrt(-np.log(cluster_powers / cluster_powers.max())) / angle_scale


    def xxxx(self) -> Tuple[np.ndarray, np.ndarray]:

        rice_factor = self._rng.normal(loc=self.rice_factor_mean, scale=self.rice_factor_std)

        cluster_delays = self._cluster_delays(rice_factor)
        cluster_powers = self._cluster_powers(cluster_delays, rice_factor)





