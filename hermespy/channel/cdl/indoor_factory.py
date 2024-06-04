# -*- coding: utf-8 -*-

from __future__ import annotations
from math import exp, log, log10
from enum import Enum
from typing import Any, Mapping, Set, Tuple, Type

import numpy as np
from h5py import Group


from hermespy.core.factory import Serializable
from .cluster_delay_lines import (
    ClusterDelayLineBase,
    ClusterDelayLineRealizationParameters,
    ClusterDelayLineSample,
    ClusterDelayLineSampleParameters,
    ClusterDelayLineRealization,
    LOSState,
)
from ..channel import ChannelSampleHook
from ..consistent import ConsistentGenerator, ConsistentRealization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class FactoryType(Enum):
    """Type of indoor factory.

    Defined in TR 138.901 v17.0.0 Table 7.2-4.
    """

    SL = 0, 10.0, 0.20
    SH = 1, 10.0, 0.20
    DL = 2, 02.0, 0.60
    DH = 3, 02.0, 0.60
    HH = 4, 05.0, 0.00

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: int, clutter_size: float, clutter_density: float) -> None:
        self.__clutter_size = clutter_size
        self.__clutter_density = clutter_density

    @property
    def clutter_size(self) -> float:
        return self.__clutter_size

    @property
    def clutter_density(self) -> float:
        return self.__clutter_density


class IndoorFactoryRealization(ClusterDelayLineRealization[LOSState]):
    """Realization of the indoor factory channel model."""

    __los_realization: ConsistentRealization
    __nlos_realization: ConsistentRealization
    __volume: float
    __surface: float
    __factory_type: FactoryType
    __clutter_height: float

    def __init__(
        self,
        expected_state: LOSState | None,
        state_realization: ConsistentRealization,
        los_realization: ConsistentRealization,
        nlos_realization: ConsistentRealization,
        parameters: ClusterDelayLineRealizationParameters,
        volume: float,
        surface: float,
        factory_type: FactoryType,
        clutter_height: float,
        sample_hooks: Set[ChannelSampleHook[ClusterDelayLineSample]],
        gain: float = 1.0,
    ) -> None:
        """
        Args:

            expected_state (O2IState | None):
                Expected large-scale state of the channel.
                If not specified, the large-scale state is randomly generated.

            state_realization (ConsistentRealization):
                Realization of a spatially consistent random number generator for the large-scale state.

            los_realization (ConsistentRealization):
                Realization of a spatially consistent random number generator for small-scale parameters in the LOS state.

            nlos_realization (ConsistentRealization):
                Realization of a spatially consistent random number generator for small-scale parameters in the NLOS state.

            parameters (ClusterDelayLineRealizationParameters):
                General parameters of the cluster delay line realization.

            volume (float):
                Volume of the modeled factory hall in :math:`\\mathrm{m}^3`.

            surface (float):
                Surface area of the modeled factory hall in :math:`\\mathrm{m}^2`.

            factory_type (FactoryType):
                Type of the factory.

            clutter_height (float):
                Height of the clutter in the factory hall in meters above the floor.

            gain (float, optional):
                Linear amplitude scaling factor if signals propagated over the channel.
        """
        # Initialize base class
        ClusterDelayLineRealization.__init__(
            self, expected_state, state_realization, parameters, sample_hooks, gain
        )

        # Initialize class attributes
        self.__los_realization = los_realization
        self.__nlos_realization = nlos_realization
        self.__volume = volume
        self.__surface = surface
        self.__factory_type = factory_type
        self.__clutter_height = clutter_height

    # Table 7.4.4-1 in TR 138.901 v17.0.0
    def _pathloss_dB(self, state: LOSState, parameters: ClusterDelayLineSampleParameters) -> float:
        PL_LOS = (
            31.84
            + 21.5 * log10(parameters.distance_3d)
            + 19 * log10(parameters.carrier_frequency / 1e9)
        )

        if state == LOSState.LOS:
            return PL_LOS

        if self.__factory_type == FactoryType.SL:
            PL_NLOS = (
                33
                + 25.5 * log10(parameters.distance_3d)
                + 20 * log10(parameters.carrier_frequency / 1e9)
            )

        elif self.__factory_type == FactoryType.DL:
            PL_NLOS = (
                18.6
                + 35.7 * log10(parameters.distance_3d)
                + 20 * log10(parameters.carrier_frequency / 1e9)
            )

        elif self.__factory_type == FactoryType.SH:
            PL_NLOS = (
                32.4
                + 23.0 * log10(parameters.distance_3d)
                + 20 * log10(parameters.carrier_frequency / 1e9)
            )

        else:  # FactoryType.DH
            PL_NLOS = (
                33.63
                + 21.9 * log10(parameters.distance_3d)
                + 20 * log10(parameters.carrier_frequency / 1e9)
            )

        return max(PL_LOS, PL_NLOS)

    def _small_scale_realization(self, state: LOSState) -> ConsistentRealization:
        if state == LOSState.LOS:
            return self.__los_realization
        else:
            return self.__nlos_realization

    def _sample_large_scale_state(
        self, state_variable_sample: float, parameters: ClusterDelayLineSampleParameters
    ) -> LOSState:
        # Implementation of TR 138.901 v17.0.0 Table 7.4.2-1

        if self.__factory_type == FactoryType.HH:
            return LOSState.LOS

        if self.__factory_type == FactoryType.SL or self.__factory_type == FactoryType.DL:
            k_subspace = -self.__factory_type.clutter_size / log(
                1 - self.__factory_type.clutter_density
            )
        else:
            k_subspace = (
                -self.__factory_type.clutter_size
                * (parameters.base_height - parameters.terminal_height)
                / (
                    log(1 - self.__factory_type.clutter_density)
                    * (self.__clutter_height - parameters.terminal_height)
                )
            )

        los_probability = exp(-parameters.distance_2d / k_subspace) if k_subspace > 0.0 else 0.0
        return LOSState.LOS if state_variable_sample < los_probability else LOSState.NLOS

    @staticmethod
    def __parameter_dependency(carrier_frequency: float, factor: float, summand: float) -> float:
        """An implementation of the frequently used equation

        .. math::

           y = a \\log_{10}(1 + f_c) + b

        Args:
            carrier_frequency (float): Carrier frequency
            factor (float): Factor scaling the logarithmic frequency dependency.
            summand (float): Added constant.

        Returns: The result.
        """

        # Note that the standard does not lower-bound the frequency for the indoor-office scenario
        # This might be an error!!!!
        fc = carrier_frequency * 1e-9
        return factor * log10(1 + fc) + summand

    # Parameters for computing the mean delay spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __delay_spread_mean: Mapping[LOSState, Tuple[float, float, float]] = {
        LOSState.LOS: (26, 14, -9.35),
        LOSState.NLOS: (30, 32, -9.44),
    }

    def _delay_spread_mean(self, state: LOSState, carrier_frequency: float) -> float:
        parameters = IndoorFactoryRealization.__delay_spread_mean[state]
        return log10(parameters[0] * self.__volume / self.__surface + parameters[1]) + parameters[2]

    # Parameters for computing the standard deviation of the delay spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __delay_spread_std: Mapping[LOSState, float] = {LOSState.LOS: 0.15, LOSState.NLOS: 0.19}

    @staticmethod
    def _delay_spread_std(state: LOSState, carrier_frequency: float) -> float:
        return IndoorFactoryRealization.__delay_spread_std[state]

    # Parameters for computing the mean angle of departure spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aod_spread_mean: Mapping[LOSState, float] = {LOSState.LOS: 1.56, LOSState.NLOS: 1.57}

    @staticmethod
    def _aod_spread_mean(state: LOSState, carrier_frequency: float) -> float:
        return IndoorFactoryRealization.__aod_spread_mean[state]

    # Parameters for computing the standard deviation of the angle of departure spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aod_spread_std: Mapping[LOSState, float] = {LOSState.LOS: 0.25, LOSState.NLOS: 0.2}

    @staticmethod
    def _aod_spread_std(state: LOSState, carrier_frequency: float) -> float:
        return IndoorFactoryRealization.__aod_spread_std[state]

    # Parameters for computing the mean angle of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aoa_spread_mean: Mapping[LOSState, Tuple[float, float]] = {
        LOSState.LOS: (-0.18, 1.78),
        LOSState.NLOS: (0.0, 1.72),
    }

    @staticmethod
    def _aoa_spread_mean(state: LOSState, carrier_frequency: float) -> float:
        mean_parameters = IndoorFactoryRealization.__aoa_spread_mean[state]
        return IndoorFactoryRealization.__parameter_dependency(carrier_frequency, *mean_parameters)

    # Parameters for computing the standard deviation of the angle of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aoa_spread_std: Mapping[LOSState, Tuple[float, float]] = {
        LOSState.LOS: (0.12, 0.2),
        LOSState.NLOS: (0.0, 0.3),
    }

    @staticmethod
    def _aoa_spread_std(state: LOSState, carrier_frequency: float) -> float:
        std_parameters = IndoorFactoryRealization.__aoa_spread_std[state]
        return IndoorFactoryRealization.__parameter_dependency(carrier_frequency, *std_parameters)

    # Parameters for computing the mean zenith of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __zoa_spread_mean: Mapping[LOSState, Tuple[float, float]] = {
        LOSState.LOS: (-0.2, 1.5),
        LOSState.NLOS: (-0.13, 1.45),
    }

    @staticmethod
    def _zoa_spread_mean(state: LOSState, carrier_frequency: float) -> float:
        mean_parameters = IndoorFactoryRealization.__zoa_spread_mean[state]
        return IndoorFactoryRealization.__parameter_dependency(carrier_frequency, *mean_parameters)

    # Parameters for computing the standard deviation of the zenith of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __zoa_spread_std: Mapping[LOSState, float] = {LOSState.LOS: 0.35, LOSState.NLOS: 0.45}

    @staticmethod
    def _zoa_spread_std(state: LOSState, carrier_frequency: float) -> float:
        return IndoorFactoryRealization.__zoa_spread_std[state]

    @staticmethod
    def _rice_factor_mean() -> float:
        return 7.0

    @staticmethod
    def _rice_factor_std() -> float:
        return 8.0

    # Delay scaling factors for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __delay_scaling: Mapping[LOSState, float] = {LOSState.LOS: 2.7, LOSState.NLOS: 3.0}

    @staticmethod
    def _delay_scaling(state: LOSState) -> float:
        return IndoorFactoryRealization.__delay_scaling[state]

    # Mean cross-polarization power ratio for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __cross_polarization_power_mean: Mapping[LOSState, float] = {
        LOSState.LOS: 12.0,
        LOSState.NLOS: 11.0,
    }

    @staticmethod
    def _cross_polarization_power_mean(state: LOSState) -> float:
        return IndoorFactoryRealization.__cross_polarization_power_mean[state]

    @staticmethod
    def _cross_polarization_power_std(state: LOSState) -> float:
        # TR 138.901 v17.0.0 Table 7.5-6
        return 6.0

    @staticmethod
    def _num_clusters(state: LOSState) -> int:
        return 25

    @staticmethod
    def _cluster_delay_spread(state: LOSState, carrier_frequency: float) -> float:
        return 0.0  # pragma: no cover

    @staticmethod
    def _cluster_aod_spread(state: LOSState) -> float:
        return 5.0

    @staticmethod
    def _cluster_aoa_spread(state: LOSState) -> float:
        return 8.0

    @staticmethod
    def _cluster_zoa_spread(state: LOSState) -> float:
        return 9.0

    # Standard deviation of the shadowing for different LOS states in dB
    # TR 138.901 v17.0.0 Table 7.5-6
    __cluster_shadowing_std: Mapping[LOSState, float] = {LOSState.LOS: 4.0, LOSState.NLOS: 3.0}

    @staticmethod
    def _cluster_shadowing_std(state: LOSState) -> float:
        return IndoorFactoryRealization.__cluster_shadowing_std[state]

    # Mean zenith of departure spread for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-11
    __zod_spread_mean: Mapping[LOSState, float] = {LOSState.LOS: 1.35, LOSState.NLOS: 1.2}

    @staticmethod
    def _zod_spread_mean(state: LOSState, parameters: ClusterDelayLineSampleParameters) -> float:
        return IndoorFactoryRealization.__zod_spread_mean[state]

    # Standard deviation of the zenith of departure spread for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-11
    __zod_spread_std: Mapping[LOSState, float] = {LOSState.LOS: 0.35, LOSState.NLOS: 0.5}

    @staticmethod
    def _zod_spread_std(state: LOSState, parameters: ClusterDelayLineSampleParameters) -> float:
        return IndoorFactoryRealization.__zod_spread_std[state]

    @staticmethod
    def _zod_offset(state: LOSState, parameters: ClusterDelayLineSampleParameters) -> float:
        # TR 138.901 v17.0.0 Table 7.5-11
        return 0.0

    def to_HDF(self, group: Group) -> None:
        ClusterDelayLineRealization.to_HDF(self, group)

        self.__los_realization.to_HDF(group.create_group("los_realization"))
        self.__nlos_realization.to_HDF(group.create_group("nlos_realization"))
        if self.expected_state is not None:
            group.attrs["expected_state"] = self.expected_state.value
        group.attrs["volume"] = self.__volume
        group.attrs["surface"] = self.__surface
        group.attrs["factory_type"] = self.__factory_type.value
        group.attrs["clutter_height"] = self.__clutter_height

    @classmethod
    def From_HDF(
        cls: Type[IndoorFactoryRealization],
        group: Group,
        parameters: ClusterDelayLineRealizationParameters,
        sample_hooks: Set[ChannelSampleHook[ClusterDelayLineSample]],
    ) -> IndoorFactoryRealization:

        state_realization = ConsistentRealization.from_HDF(group["state_realization"])
        los_realization = ConsistentRealization.from_HDF(group["los_realization"])
        nlos_realization = ConsistentRealization.from_HDF(group["nlos_realization"])
        if "expected_state" in group.attrs:
            expected_state = LOSState(group.attrs["expected_state"])
        else:
            expected_state = None
        volume = group.attrs["volume"]
        surface = group.attrs["surface"]
        factory_type = FactoryType(group.attrs["factory_type"])  # type: ignore[call-arg]
        clutter_height = group.attrs["clutter_height"]
        gain = group.attrs["gain"] if "gain" in group.attrs else 1.0

        return IndoorFactoryRealization(
            expected_state,
            state_realization,
            los_realization,
            nlos_realization,
            parameters,
            volume,
            surface,
            factory_type,
            clutter_height,
            sample_hooks,
            gain,
        )


class IndoorFactory(ClusterDelayLineBase[IndoorFactoryRealization, LOSState], Serializable):
    """3GPP cluster delay line preset modeling an indoor factory scenario."""

    yaml_tag = "IndoorFactory"
    __volume: float  # Hall volume in m3
    __surface: float  # Total surface hall area in m2 (walls/floor/ceiling)
    __factory_type: FactoryType
    __clutter_height: float

    def __init__(
        self,
        volume: float,
        surface: float,
        factory_type: FactoryType,
        clutter_height: float = 0.0,
        gain: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Args:

            volume (float):
                Hall volume in :math:`\\mathrm{m}^3`.

            surface (float):
                Total surface hall area in :math:`\\mathrm{m}^2`. (walls/floor/ceiling).

            factory_type (FactoryType):
                Type of the factory.

            clutter_height (float, optional):
                Height of the clutter in the factory hall in meters above the floor.
                Zero by default, meaning virtually no clutter.

            gain (float, optional):
                Linear power gain factor a signal experiences when being propagated over this realization.
                :math:`1.0` by default.

            \**kwargs:
                Additional arguments passed to the base class.
        """

        # Initialize base class
        ClusterDelayLineBase.__init__(self, gain, **kwargs)

        # Initialize class attributes
        self.volume = volume
        self.surface = surface
        self.factory_type = factory_type
        self.clutter_height = clutter_height

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

    @property
    def factory_type(self) -> FactoryType:
        """Assumed type of factory."""

        return self.__factory_type

    @factory_type.setter
    def factory_type(self, value: FactoryType) -> None:
        self.__factory_type = value

    @property
    def clutter_height(self) -> float:
        """Cluter height in m.

        Denoted by :math:`h_c` within the respective equations.
        Should be lower than ceiling height and in between zero and 10m.
        """
        return self.__clutter_height

    @clutter_height.setter
    def clutter_height(self, value: float) -> None:
        if value < 0.0 or value > 10.0:
            raise ValueError("Clutter height should be in the interval 0-10m")
        self.__clutter_height = value

    @property
    def max_num_clusters(self) -> int:
        return 25

    @property
    def max_num_rays(self) -> int:
        return 20

    @property
    def _large_scale_correlations(self) -> np.ndarray:
        # Large scale cross correlations
        # TR 138.901 v17.0.0 Table 7.5-6
        return np.array(
            [
                #    LOS     NLOS
                [+0.00, +0.00],  # 0: ASD vs DS
                [+0.00, +0.00],  # 1: ASA vs DS
                [+0.00, +0.00],  # 2: ASA VS SF
                [+0.00, +0.00],  # 3: ASD vs SF
                [+0.00, +0.00],  # 4: DS vs SF
                [+0.00, +0.00],  # 5: ASD vs ASA
                [-0.50, +0.00],  # 6: ASD vs K
                [+0.00, +0.00],  # 7: ASA vs K
                [-0.70, +0.00],  # 8: DS vs K
                [+0.00, +0.00],  # 9: SF vs K
                [+0.00, +0.00],  # 10: ZSD vs SF
                [+0.00, +0.00],  # 11: ZSA vs SF
                [+0.00, +0.00],  # 12: ZSD vs K
                [+0.00, +0.00],  # 13: ZSA vs K
                [+0.00, +0.00],  # 14: ZSD vs DS
                [+0.00, +0.00],  # 15: ZSA vs DS
                [+0.00, +0.00],  # 16: ZSD vs ASD
                [+0.00, +0.00],  # 17: ZSA vs ASD
                [+0.00, +0.00],  # 18: ZSD vs ASA
                [+0.00, +0.00],  # 19: ZSA vs ASA
                [+0.00, +0.00],  # 20: ZSD vs ZSA
            ],
            dtype=np.float_,
        ).T

    def _initialize_realization(
        self,
        state_generator: ConsistentGenerator,
        parameter_generator: ConsistentGenerator,
        parameters: ClusterDelayLineRealizationParameters,
    ) -> IndoorFactoryRealization:

        # Generate realizations for each large scale state
        # TR 138.901 v17.0.0 Table 7.6.3.1-2
        state_realization = state_generator.realize(0.5 * self.factory_type.clutter_size)
        los_realization = parameter_generator.realize(10.0)
        nlos_realization = parameter_generator.realize(10.0)

        return IndoorFactoryRealization(
            self.expected_state,
            state_realization,
            los_realization,
            nlos_realization,
            parameters,
            self.volume,
            self.surface,
            self.factory_type,
            self.clutter_height,
            self.sample_hooks,
            self.gain,
        )

    def _recall_specific_realization(
        self, group: Group, parameters: ClusterDelayLineRealizationParameters
    ) -> IndoorFactoryRealization:
        return IndoorFactoryRealization.From_HDF(group, parameters, self.sample_hooks)
