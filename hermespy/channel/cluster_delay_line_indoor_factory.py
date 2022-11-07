# -*- coding: utf-8 -*-
"""
============================================
3GPP Cluster Delay Line Indoor Factory Model
============================================

Implements several parameter sets defined within the 3GPP standard modeling specific scenarios.
"""

from abc import ABCMeta
from math import log10
from typing import Any

from ..core.factory import Serializable
from .cluster_delay_lines import ClusterDelayLineBase

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class IndoorFactoryBase(ClusterDelayLineBase, metaclass=ABCMeta):
    """Indoor Factory Cluster Delay Line Model Base."""

    __volume: float     # Hall volume in m3
    __surface: float    # Total surface hall area in m2 (walls/floor/ceiling)

    def __init__(self,
                 volume: float,
                 surface: float,
                 **kwargs: Any) -> None:
        """
        Args:

            volume (float):
                Hall volume in m3.

            surface (float):
                Total surface hall area in m2 (walls/floor/ceiling).
        """

        self.volume = volume
        self.surface = surface
        ClusterDelayLineBase.__init__(self, **kwargs)

    @property
    def volume(self) -> float:
        """Hall volume.

        Returns:

            float: Volume in m3.

        Raises:
            ValueError:
                For volumes smaller or equal to zero.
        """

        return self.__volume

    @volume.setter
    def volume(self, value: float) -> None:

        if value <= 0.:
            raise ValueError("Hall volume must be greater than zero")

        self.__volume = value

    @property
    def surface(self) -> float:
        """Hall surface area.

        Returns:

            float: Surface area in m2.

        Raises:
            ValueError:
                For surfaces areas smaller or equal to zero.
        """

        return self.__surface

    @surface.setter
    def surface(self, value: float) -> None:

        if value <= 0.:
            raise ValueError("Hall surface area must be greater than zero")

        self.__surface = value


class IndoorFactoryLineOfSight(IndoorFactoryBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Indoor-Factory Model."""

    yaml_tag = u'IndoorFactoryLOS'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return True

    @property
    def delay_spread_mean(self) -> float:
        return log10(26 * self.volume / self.surface + 14) - 9.35

    @property
    def delay_spread_std(self) -> float:
        return .15

    @property
    def aod_spread_mean(self) -> float:
        return 1.56

    @property
    def aod_spread_std(self) -> float:
        return .25

    @property
    def aoa_spread_mean(self) -> float:
        return -.18 * log10(1 + self._center_frequency * 1e-9) + 1.78

    @property
    def aoa_spread_std(self) -> float:
        return .12 * log10(1 + self._center_frequency * 1e-9) + .2

    @property
    def zoa_spread_mean(self) -> float:
        return -.2 * log10(1 + self._center_frequency * 1e-9) + 1.5

    @property
    def zoa_spread_std(self) -> float:
        return .35

    @property
    def zod_spread_mean(self) -> float:
        return 1.35

    @property
    def zod_spread_std(self) -> float:
        return .35

    @property
    def zod_offset(self) -> float:
        return 0.

    @property
    def rice_factor_mean(self) -> float:
        return 7.

    @property
    def rice_factor_std(self) -> float:
        return 8.

    @property
    def delay_scaling(self) -> float:
        return 2.7

    @property
    def cross_polarization_power_mean(self) -> float:
        return 12.

    @property
    def cross_polarization_power_std(self) -> float:
        return 6.

    @property
    def num_clusters(self) -> int:
        return 25

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.

    @property
    def cluster_aod_spread(self) -> float:
        return 5.

    @property
    def cluster_aoa_spread(self) -> float:
        return 8.

    @property
    def cluster_zoa_spread(self) -> float:
        return 9.

    @property
    def cluster_shadowing_std(self) -> float:
        return 4.

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))


class IndoorFactoryNoLineOfSight(IndoorFactoryBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Indoor-Factory Model."""

    yaml_tag = u'IndoorFactoryNLOS'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return log10(30 * self.volume / self.surface + 32) - 9.44

    @property
    def delay_spread_std(self) -> float:
        return .19

    @property
    def aod_spread_mean(self) -> float:
        return 1.57

    @property
    def aod_spread_std(self) -> float:
        return .2

    @property
    def aoa_spread_mean(self) -> float:
        return 1.72

    @property
    def aoa_spread_std(self) -> float:
        return .3

    @property
    def zoa_spread_mean(self) -> float:
        return -.13 * log10(1 + self._center_frequency * 1e-9) + 1.45

    @property
    def zoa_spread_std(self) -> float:
        return .45

    @property
    def zod_spread_mean(self) -> float:
        return 1.2

    @property
    def zod_spread_std(self) -> float:
        return .55

    @property
    def zod_offset(self) -> float:
        return 0.

    @property
    def rice_factor_mean(self) -> float:
        return 0.

    @property
    def rice_factor_std(self) -> float:
        return 0.

    @property
    def delay_scaling(self) -> float:
        return 3.

    @property
    def cross_polarization_power_mean(self) -> float:
        return 11.

    @property
    def cross_polarization_power_std(self) -> float:
        return 6.

    @property
    def num_clusters(self) -> int:
        return 25

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.

    @property
    def cluster_aod_spread(self) -> float:
        return 5.

    @property
    def cluster_aoa_spread(self) -> float:
        return 8.

    @property
    def cluster_zoa_spread(self) -> float:
        return 9.

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))
