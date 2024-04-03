# -*- coding: utf-8 -*-

from math import atan

import numpy as np

from ..core.factory import Serializable
from .cluster_delay_lines import ClusterDelayLineBase

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RuralMacrocellsLineOfSight(ClusterDelayLineBase, Serializable):
    """3GPP cluster delay line preset modeling a rural scenario with direct line of sight
    between the linked wireless devices.

    Refer to the :footcite:t:`3GPP:TR38901` for detailed information.

    The following minimal example outlines how to configure the channel model
    within the context of a :doc:`simulation.simulation.Simulation`:

    .. literalinclude:: ../scripts/examples/channel_cdl_rural_macrocells_los.py
       :language: python
       :linenos:
       :lines: 12-40
    """

    yaml_tag = "RMaLOS"
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return True

    @property
    def delay_spread_mean(self) -> float:
        return -7.49

    @property
    def delay_spread_std(self) -> float:
        return 0.55

    @property
    def aod_spread_mean(self) -> float:
        return 0.9

    @property
    def aod_spread_std(self) -> float:
        return 0.38

    @property
    def aoa_spread_mean(self) -> float:
        return 1.52

    @property
    def aoa_spread_std(self) -> float:
        return 0.24

    @property
    def zoa_spread_mean(self) -> float:
        return 0.47

    @property
    def zoa_spread_std(self) -> float:
        return 0.4

    @property
    def zod_spread_mean(self) -> float:
        device_distance = np.linalg.norm(self.beta_device.position - self.alpha_device.position, 2)
        terminal_height = abs(self.alpha_device.position[2] - self.beta_device.position[2])

        return max(-1, -17e-5 * device_distance - 0.01 * (terminal_height - 1.5) + 0.22)

    @property
    def zod_spread_std(self) -> float:
        return 0.34

    @property
    def zod_offset(self) -> float:
        return 0.0

    @property
    def rice_factor_mean(self) -> float:
        return 7.0

    @property
    def rice_factor_std(self) -> float:
        return 4.0

    @property
    def delay_scaling(self) -> float:
        return 3.8

    @property
    def cross_polarization_power_mean(self) -> float:
        return 12.0

    @property
    def cross_polarization_power_std(self) -> float:
        return 4.0

    @property
    def num_clusters(self) -> int:
        return 11

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.0

    @property
    def cluster_aod_spread(self) -> float:
        return 2.0

    @property
    def cluster_aoa_spread(self) -> float:
        return 3.0

    @property
    def cluster_zoa_spread(self) -> float:
        return 3.0

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.0

    @property
    def _center_frequency(self) -> float:  # pragma: no cover
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))  # type: ignore


class RuralMacrocellsNoLineOfSight(ClusterDelayLineBase, Serializable):
    """3GPP cluster delay line preset modeling a rural scenario without direct line of sight
    between the linked wireless devices.

    Refer to the :footcite:t:`3GPP:TR38901` for detailed information.

    The following minimal example outlines how to configure the channel model
    within the context of a :doc:`simulation.simulation.Simulation`:

    .. literalinclude:: ../scripts/examples/channel_cdl_rural_macrocells_nlos.py
       :language: python
       :linenos:
       :lines: 12-40
    """

    yaml_tag = "RMaLOS"
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return -7.43

    @property
    def delay_spread_std(self) -> float:
        return 0.48

    @property
    def aod_spread_mean(self) -> float:
        return 0.95

    @property
    def aod_spread_std(self) -> float:
        return 0.45

    @property
    def aoa_spread_mean(self) -> float:
        return 1.52

    @property
    def aoa_spread_std(self) -> float:
        return 0.13

    @property
    def zoa_spread_mean(self) -> float:
        return 0.58

    @property
    def zoa_spread_std(self) -> float:
        return 0.37

    @property
    def zod_spread_mean(self) -> float:
        device_distance = np.linalg.norm(self.beta_device.position - self.alpha_device.position, 2)
        terminal_height = abs(self.alpha_device.position[2] - self.beta_device.position[2])

        return max(-1, -19e-5 * device_distance - 0.01 * (terminal_height - 1.5) + 0.28)

    @property
    def zod_spread_std(self) -> float:
        return 0.3

    @property
    def zod_offset(self) -> float:
        device_distance = np.linalg.norm(self.beta_device.position - self.alpha_device.position, 2)

        return atan((35 - 0.35) / device_distance) - atan((35 - 1.5) / device_distance)

    @property
    def rice_factor_mean(self) -> float:
        return 0.0

    @property
    def rice_factor_std(self) -> float:
        return 0.0

    @property
    def delay_scaling(self) -> float:
        return 1.7

    @property
    def cross_polarization_power_mean(self) -> float:
        return 7.0

    @property
    def cross_polarization_power_std(self) -> float:
        return 3.0

    @property
    def num_clusters(self) -> int:
        return 10

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.0

    @property
    def cluster_aod_spread(self) -> float:
        return 2.0

    @property
    def cluster_aoa_spread(self) -> float:
        return 3.0

    @property
    def cluster_zoa_spread(self) -> float:
        return 3.0

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.0

    @property
    def _center_frequency(self) -> float:  # pragma: no cover
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))  # type: ignore


class RuralMacrocellsOutsideToInside(ClusterDelayLineBase, Serializable):
    """3GPP cluster delay line preset modeling a rural scenario with
    the linked wireless devices being outside and inside a building, respectively.

    Refer to the :footcite:t:`3GPP:TR38901` for detailed information.

    The following minimal example outlines how to configure the channel model
    within the context of a :doc:`simulation.simulation.Simulation`:

    .. literalinclude:: ../scripts/examples/channel_cdl_rural_macrocells_o2i.py
       :language: python
       :linenos:
       :lines: 12-40
    """

    yaml_tag = "RMaO2I"
    """YAML serialization tag."""

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return -7.47

    @property
    def delay_spread_std(self) -> float:
        return 0.24

    @property
    def aod_spread_mean(self) -> float:
        return 0.67

    @property
    def aod_spread_std(self) -> float:
        return 0.18

    @property
    def aoa_spread_mean(self) -> float:
        return 1.66

    @property
    def aoa_spread_std(self) -> float:
        return 0.21

    @property
    def zoa_spread_mean(self) -> float:
        return 0.93

    @property
    def zoa_spread_std(self) -> float:
        return 0.22

    @property
    def zod_spread_mean(self) -> float:
        device_distance = np.linalg.norm(self.beta_device.position - self.alpha_device.position, 2)
        terminal_height = abs(self.alpha_device.position[2] - self.beta_device.position[2])

        return max(-1, -19e-5 * device_distance - 0.01 * (terminal_height - 1.5) + 0.28)

    @property
    def zod_spread_std(self) -> float:
        return 0.3

    @property
    def zod_offset(self) -> float:
        device_distance = np.linalg.norm(self.beta_device.position - self.alpha_device.position, 2)

        return atan((35 - 0.35) / device_distance) - atan((35 - 1.5) / device_distance)

    @property
    def rice_factor_mean(self) -> float:
        return 0.0

    @property
    def rice_factor_std(self) -> float:
        return 0.0

    @property
    def delay_scaling(self) -> float:
        return 1.7

    @property
    def cross_polarization_power_mean(self) -> float:
        return 7.0

    @property
    def cross_polarization_power_std(self) -> float:
        return 3.0

    @property
    def num_clusters(self) -> int:
        return 10

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.0

    @property
    def cluster_aod_spread(self) -> float:
        return 2.0

    @property
    def cluster_aoa_spread(self) -> float:
        return 3.0

    @property
    def cluster_zoa_spread(self) -> float:
        return 3.0

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.0

    @property
    def _center_frequency(self) -> float:  # pragma: no cover
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))  # type: ignore
