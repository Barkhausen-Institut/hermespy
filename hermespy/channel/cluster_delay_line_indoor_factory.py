# -*- coding: utf-8 -*-

from math import log10
from typing import Any

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


class IndoorFactoryBase(ClusterDelayLineBase):
    """Indoor Factory Cluster Delay Line Model Base."""

    __volume: float  # Hall volume in m3
    __surface: float  # Total surface hall area in m2 (walls/floor/ceiling)

    def __init__(self, volume: float, surface: float, alpha_device=None, beta_device=None, gain: float = 1.0, **kwargs: Any) -> None:
        """
        Args:

            volume (float):
                Hall volume in :math:`\\mathrm{m}^3`.

            surface (float):
                Total surface hall area in :math:`\\mathrm{m}^2`. (walls/floor/ceiling).

            alpha_device (SimulatedDevice, optional):
                First device linked by the :class:`.ClusterDelayLine` instance.

            beta_device (SimulatedDevice, optional):
                Second device linked by the :class:`.ClusterDelayLine` instance.

            gain (float, optional):
                Linear power gain factor a signal experiences when being propagated over this realization.
                :math:`1.0` by default.
        """

        # Initialize base class
        ClusterDelayLineBase.__init__(self, alpha_device, beta_device, gain, **kwargs)

        # Initialize class attributes
        self.volume = volume
        self.surface = surface

    @property
    def volume(self) -> float:
        """Assumed factory hall volume in :math:`\\mathrm{m}^3`.

        Raises:

            ValueError: For values smaller or equal to zero.
        """

        return self.__volume

    @volume.setter
    def volume(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Hall volume must be greater than zero")

        self.__volume = value

    @property
    def surface(self) -> float:
        """Assumed factory hall surface in :math:`\\mathrm{m}^2`.

        Raises:

            ValueError: For values smaller or equal to zero.
        """

        return self.__surface

    @surface.setter
    def surface(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Hall surface area must be greater than zero")

        self.__surface = value


class IndoorFactoryLineOfSight(IndoorFactoryBase, Serializable):
    """3GPP cluster delay line preset modeling an indoor factory scenario with direct line of sight
    between the linked wireless devices.

    Refer to the :footcite:t:`3GPP:TR38901` for detailed information.

    The following minimal example outlines how to configure the channel model
    within the context of a :doc:`simulation.simulation.Simulation`:

    .. literalinclude:: ../scripts/examples/channel_cdl_indoor_factory_los.py
       :language: python
       :linenos:
       :lines: 12-40
    """

    yaml_tag = "IndoorFactoryLOS"

    @property
    def line_of_sight(self) -> bool:
        return True

    @property
    def delay_spread_mean(self) -> float:
        return log10(26 * self.volume / self.surface + 14) - 9.35

    @property
    def delay_spread_std(self) -> float:
        return 0.15

    @property
    def aod_spread_mean(self) -> float:
        return 1.56

    @property
    def aod_spread_std(self) -> float:
        return 0.25

    @property
    def aoa_spread_mean(self) -> float:
        return -0.18 * log10(1 + self._center_frequency * 1e-9) + 1.78

    @property
    def aoa_spread_std(self) -> float:
        return 0.12 * log10(1 + self._center_frequency * 1e-9) + 0.2

    @property
    def zoa_spread_mean(self) -> float:
        return -0.2 * log10(1 + self._center_frequency * 1e-9) + 1.5

    @property
    def zoa_spread_std(self) -> float:
        return 0.35

    @property
    def zod_spread_mean(self) -> float:
        return 1.35

    @property
    def zod_spread_std(self) -> float:
        return 0.35

    @property
    def zod_offset(self) -> float:
        return 0.0

    @property
    def rice_factor_mean(self) -> float:
        return 7.0

    @property
    def rice_factor_std(self) -> float:
        return 8.0

    @property
    def delay_scaling(self) -> float:
        return 2.7

    @property
    def cross_polarization_power_mean(self) -> float:
        return 12.0

    @property
    def cross_polarization_power_std(self) -> float:
        return 6.0

    @property
    def num_clusters(self) -> int:
        return 25

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.0

    @property
    def cluster_aod_spread(self) -> float:
        return 5.0

    @property
    def cluster_aoa_spread(self) -> float:
        return 8.0

    @property
    def cluster_zoa_spread(self) -> float:
        return 9.0

    @property
    def cluster_shadowing_std(self) -> float:
        return 4.0

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))  # type: ignore


class IndoorFactoryNoLineOfSight(IndoorFactoryBase, Serializable):
    """3GPP cluster delay line preset modeling an indoor factory scenario without direct line of sight
    between the linked wireless devices.

    Refer to the :footcite:t:`3GPP:TR38901` for detailed information.

    The following minimal example outlines how to configure the channel model
    within the context of a :doc:`simulation.simulation.Simulation`:

    .. literalinclude:: ../scripts/examples/channel_cdl_indoor_factory_nlos.py
       :language: python
       :linenos:
       :lines: 12-40
    """

    yaml_tag = "IndoorFactoryNLOS"

    @property
    def line_of_sight(self) -> bool:
        return False

    @property
    def delay_spread_mean(self) -> float:
        return log10(30 * self.volume / self.surface + 32) - 9.44

    @property
    def delay_spread_std(self) -> float:
        return 0.19

    @property
    def aod_spread_mean(self) -> float:
        return 1.57

    @property
    def aod_spread_std(self) -> float:
        return 0.2

    @property
    def aoa_spread_mean(self) -> float:
        return 1.72

    @property
    def aoa_spread_std(self) -> float:
        return 0.3

    @property
    def zoa_spread_mean(self) -> float:
        return -0.13 * log10(1 + self._center_frequency * 1e-9) + 1.45

    @property
    def zoa_spread_std(self) -> float:
        return 0.45

    @property
    def zod_spread_mean(self) -> float:
        return 1.2

    @property
    def zod_spread_std(self) -> float:
        return 0.55

    @property
    def zod_offset(self) -> float:
        return 0.0

    @property
    def rice_factor_mean(self) -> float:
        return 0.0

    @property
    def rice_factor_std(self) -> float:
        return 0.0

    @property
    def delay_scaling(self) -> float:
        return 3.0

    @property
    def cross_polarization_power_mean(self) -> float:
        return 11.0

    @property
    def cross_polarization_power_std(self) -> float:
        return 6.0

    @property
    def num_clusters(self) -> int:
        return 25

    @property
    def num_rays(self) -> int:
        return 20

    @property
    def cluster_delay_spread(self) -> float:
        return 0.0

    @property
    def cluster_aod_spread(self) -> float:
        return 5.0

    @property
    def cluster_aoa_spread(self) -> float:
        return 8.0

    @property
    def cluster_zoa_spread(self) -> float:
        return 9.0

    @property
    def cluster_shadowing_std(self) -> float:
        return 3.0

    @property
    def _center_frequency(self) -> float:
        return max(6e9, ClusterDelayLineBase._center_frequency.fget(self))  # type: ignore
