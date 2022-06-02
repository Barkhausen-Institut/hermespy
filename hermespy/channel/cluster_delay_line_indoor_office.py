# -*- coding: utf-8 -*-
"""
===========================================
3GPP Cluster Delay Line Indoor Office Model
===========================================

Implements several parameter sets defined within the 3GPP standard modeling specific scenarios.
"""

from abc import ABCMeta
from math import log10

import numpy as np

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


class IndoorOfficeLineOfSight(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Indoor-Office Model."""

    yaml_tag = u'IndoorOfficeLOS'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return True

    @property
    def delay_spread_mean(self) -> float:
        return -.01 * log10(1 + self._center_frequency * 1e-9) - 7.692

    @property
    def delay_spread_std(self) -> float:
        return .18

    @property
    def aod_spread_mean(self) -> float:
        return 1.6

    @property
    def aod_spread_std(self) -> float:
        return .18

    @property
    def aoa_spread_mean(self) -> float:
        return -.19 * log10(1 + self._center_frequency * 1e-9) + 1.781

    @property
    def aoa_spread_std(self) -> float:
        return .12 * log10(1 + self._center_frequency * 1e-9) + .119

    @property
    def zoa_spread_mean(self) -> float:
        return -.26 * log10(1 + self._center_frequency * 1e-9) + 1.44

    @property
    def zoa_spread_std(self) -> float:
        return -.04 * log10(1 + self._center_frequency * 1e-9) + .264

    @property
    def zod_spread_mean(self) -> float:
        return -1.43 * log10(1 + self._center_frequency) + 2.228

    @property
    def zod_spread_std(self) -> float:
        return .13 * log10(1 + self._center_frequency) + .3

    @property
    def zod_offset(self) -> float:
        return 0.

    @property
    def rice_factor_mean(self) -> float:
        return 7.

    @property
    def rice_factor_std(self) -> float:
        return 4.

    @property
    def delay_scaling(self) -> float:
        return 3.6

    @property
    def cross_polarization_power_mean(self) -> float:
        return 11.

    @property
    def cross_polarization_power_std(self) -> float:
        return 4.

    @property
    def num_clusters(self) -> int:
        return 15

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
        return 6.

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))


class IndoorOfficeNoLineOfSight(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Indoor-Office Model."""

    yaml_tag = u'IndoorOfficeNLOS'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return -.28 * log10(1 + self._center_frequency * 1e-9) - 7.173

    @property
    def delay_spread_std(self) -> float:
        return .1 * log10(1 + self._center_frequency * 1e-9) - .055

    @property
    def aod_spread_mean(self) -> float:
        return 1.62

    @property
    def aod_spread_std(self) -> float:
        return .25

    @property
    def aoa_spread_mean(self) -> float:
        return -.11 * log10(1 + self._center_frequency * 1e-9) + 1.863

    @property
    def aoa_spread_std(self) -> float:
        return .12 * log10(1 + self._center_frequency * 1e-9) + .119

    @property
    def zoa_spread_mean(self) -> float:
        return -.15 * log10(1 + self._center_frequency * 1e-9) + 1.387

    @property
    def zoa_spread_std(self) -> float:
        return -.09 * log10(1 + self._center_frequency * 1e-9) + .746

    @property
    def zod_spread_mean(self) -> float:
        return 1.08

    @property
    def zod_spread_std(self) -> float:
        return .36

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
        return 3

    @property
    def cross_polarization_power_mean(self) -> float:
        return 10.

    @property
    def cross_polarization_power_std(self) -> float:
        return 4.

    @property
    def num_clusters(self) -> int:
        return 19

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
        return 11.

    @property
    def cluster_zoa_spread(self) -> float:
        return 9.

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))
