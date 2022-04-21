# -*- coding: utf-8 -*-
"""
===========================
3GPP Rural Macrocells Model
===========================

Implements several parameter sets defined within the 3GPP standard modeling specific scenarios.
"""

from math import atan

import numpy as np

from ..core.factory import Serializable
from .cluster_delay_lines import ClusterDelayLineBase

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RuralMacrocellsLineOfSight(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Rural Macrocells Model."""

    yaml_tag = u'RMaLOS'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return True

    @property
    def delay_spread_mean(self) -> float:
        return -7.49

    @property
    def delay_spread_std(self) -> float:
        return .55

    @property
    def aod_spread_mean(self) -> float:
        return .9

    @property
    def aod_spread_std(self) -> float:
        return .38

    @property
    def aoa_spread_mean(self) -> float:
        return 1.52

    @property
    def aoa_spread_std(self) -> float:
        return .24

    @property
    def zoa_spread_mean(self) -> float:
        return .47

    @property
    def zoa_spread_std(self) -> float:
        return .4

    @property
    def zod_spread_mean(self) -> float:

        device_distance = np.linalg.norm(self.receiver.position - self.transmitter.position, 2)
        terminal_height = abs(self.transmitter.position[2] - self.receiver.position[2])

        return max(-1, -17e-5 * device_distance - .01 * (terminal_height - 1.5) + .22)

    @property
    def zod_spread_std(self) -> float:
        return .34

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
        return 3.8

    @property
    def cross_polarization_power_mean(self) -> float:
        return 12.

    @property
    def cross_polarization_power_std(self) -> float:
        return 4.

    @property
    def num_clusters(self) -> int:
        return 11

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.

    @property
    def cluster_aod_spread(self) -> float:
        return 2.

    @property
    def cluster_aoa_spread(self) -> float:
        return 3.

    @property
    def cluster_zoa_spread(self) -> float:
        return 3.

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))


class RuralMacrocellsNoLineOfSight(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Urban Macrocells Model."""

    yaml_tag = u'RMaLOS'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return -7.43

    @property
    def delay_spread_std(self) -> float:
        return .48

    @property
    def aod_spread_mean(self) -> float:
        return .95

    @property
    def aod_spread_std(self) -> float:
        return .45

    @property
    def aoa_spread_mean(self) -> float:
        return 1.52

    @property
    def aoa_spread_std(self) -> float:
        return .13

    @property
    def zoa_spread_mean(self) -> float:
        return .58

    @property
    def zoa_spread_std(self) -> float:
        return .37

    @property
    def zod_spread_mean(self) -> float:

        device_distance = np.linalg.norm(self.receiver.position - self.transmitter.position, 2)
        terminal_height = abs(self.transmitter.position[2] - self.receiver.position[2])

        return max(-1, -19e-5 * device_distance - .01 * (terminal_height - 1.5) + .28)

    @property
    def zod_spread_std(self) -> float:
        return .3

    @property
    def zod_offset(self) -> float:

        device_distance = np.linalg.norm(self.receiver.position - self.transmitter.position, 2)

        return atan((35 - .35) / device_distance) - atan((35 - 1.5) / device_distance)

    @property
    def rice_factor_mean(self) -> float:
        return 0.

    @property
    def rice_factor_std(self) -> float:
        return 0.

    @property
    def delay_scaling(self) -> float:
        return 1.7

    @property
    def cross_polarization_power_mean(self) -> float:
        return 7.

    @property
    def cross_polarization_power_std(self) -> float:
        return 3.

    @property
    def num_clusters(self) -> int:
        return 10

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.

    @property
    def cluster_aod_spread(self) -> float:
        return 2.

    @property
    def cluster_aoa_spread(self) -> float:
        return 3.

    @property
    def cluster_zoa_spread(self) -> float:
        return 3.

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))


class RuralMacrocellsOutsideToInside(ClusterDelayLineBase, Serializable):
    """Parameter Preset for the 3GPP Cluster Delay Line Urban Macrocells Model."""

    yaml_tag = u'RMaO2I'
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return -7.47

    @property
    def delay_spread_std(self) -> float:
        return .24

    @property
    def aod_spread_mean(self) -> float:
        return .67

    @property
    def aod_spread_std(self) -> float:
        return .18

    @property
    def aoa_spread_mean(self) -> float:
        return 1.66

    @property
    def aoa_spread_std(self) -> float:
        return .21

    @property
    def zoa_spread_mean(self) -> float:
        return .93

    @property
    def zoa_spread_std(self) -> float:
        return .22

    @property
    def zod_spread_mean(self) -> float:

        device_distance = np.linalg.norm(self.receiver.position - self.transmitter.position, 2)
        terminal_height = abs(self.transmitter.position[2] - self.receiver.position[2])

        return max(-1, -19e-5 * device_distance - .01 * (terminal_height - 1.5) + .28)

    @property
    def zod_spread_std(self) -> float:
        return .3

    @property
    def zod_offset(self) -> float:

        device_distance = np.linalg.norm(self.receiver.position - self.transmitter.position, 2)

        return atan((35 - .35) / device_distance) - atan((35 - 1.5) / device_distance)

    @property
    def rice_factor_mean(self) -> float:
        return 0.

    @property
    def rice_factor_std(self) -> float:
        return 0.

    @property
    def delay_scaling(self) -> float:
        return 1.7

    @property
    def cross_polarization_power_mean(self) -> float:
        return 7.

    @property
    def cross_polarization_power_std(self) -> float:
        return 3.

    @property
    def num_clusters(self) -> int:
        return 10

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.

    @property
    def cluster_aod_spread(self) -> float:
        return 2.

    @property
    def cluster_aoa_spread(self) -> float:
        return 3.

    @property
    def cluster_zoa_spread(self) -> float:
        return 3.

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))
