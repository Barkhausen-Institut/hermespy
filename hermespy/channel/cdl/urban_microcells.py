# -*- coding: utf-8 -*-

from __future__ import annotations
from math import log10
from typing import Mapping, Set, Tuple
from typing_extensions import override

import numpy as np

from hermespy.core import DeserializationProcess, SerializationProcess
from .cluster_delay_lines import (
    ClusterDelayLineRealization,
    ClusterDelayLineBase,
    ClusterDelayLineRealizationParameters,
    ClusterDelayLineSample,
    ClusterDelayLineSampleParameters,
    O2IState,
)
from ..channel import ChannelSampleHook
from ..consistent import ConsistentGenerator, ConsistentRealization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class UrbanMicrocellsRealization(ClusterDelayLineRealization[O2IState]):
    """Realization of an urban street canyon cluster delay line model."""

    def __init__(
        self,
        expected_state: O2IState | None,
        state_realization: ConsistentRealization,
        los_realization: ConsistentRealization,
        nlos_realization: ConsistentRealization,
        o2i_realization: ConsistentRealization,
        parameters: ClusterDelayLineRealizationParameters,
        sample_hooks: Set[ChannelSampleHook[ClusterDelayLineSample]],
        gain: float = 1.0,
    ) -> None:
        """
        Args:

            expected_state:
                Expected large-scale state of the channel.
                If not specified, the large-scale state is randomly generated.

            state_realization:
                Realization of a spatially consistent random number generator for the large-scale state.

            los_realization:
                Realization of a spatially consistent random number generator for small-scale parameters in the LOS state.

            nlos_realization:
                Realization of a spatially consistent random number generator for small-scale parameters in the NLOS state.

            o2i_realization:
                Realization of a spatially consistent random number generator for small-scale parameters in the O2I state.

            parameters:
                General parameters of the cluster delay line realization.

            sample_hooks:
                Hooks to be called when a channel sample is generated.

            gain:
                Linear amplitude scaling factor if signals propagated over the channel.
        """

        # Initialize base class
        ClusterDelayLineRealization.__init__(
            self, expected_state, state_realization, parameters, sample_hooks, gain
        )

        # Initialize class attributes
        self.__los_realization = los_realization
        self.__nlos_realization = nlos_realization
        self.__o2i_realization = o2i_realization

    # Table 7.4.4-1 in TR 138.901 v17.0.0
    @staticmethod
    def _pathloss_dB(
        state: O2IState, parameters: ClusterDelayLineSampleParameters
    ) -> float:  # pragma: no cover

        if state == O2IState.O2I:
            return 0.0

        h_BS = 10.0  # Height of the base station in meters
        h_UT = max(1.5, min(22.5, parameters.terminal_height))  # Height of the terminal in meters

        # Note 1 in Table 7.4.4-1 of TR 138.901 v17.0.0
        breakpoint_distance = (
            4 * (h_BS - 1) * (h_UT - 1) * parameters.carrier_frequency * 1e-8 / 3.0
        )

        if parameters.distance_2d < breakpoint_distance:
            PL_LOS = (
                32.4
                + 21 * log10(parameters.distance_3d)
                + 20 * log10(parameters.carrier_frequency * 1e-9)
            )
        else:
            PL_LOS = (
                32.4
                + 40 * log10(parameters.distance_3d)
                + 20 * log10(parameters.carrier_frequency * 1e-9)
                - 9.5 * log10(breakpoint_distance**2 + (h_BS - h_UT) ** 2)
            )

        if state == O2IState.LOS:
            return PL_LOS

        PL_NLOS = (
            35.3 * log10(parameters.distance_3d)
            + 22.4
            + 21.3 * log10(parameters.carrier_frequency * 1e-9)
            - 0.3 * (h_UT - 1.5)
        )
        return max(PL_LOS, PL_NLOS)

    def _small_scale_realization(self, state: O2IState) -> ConsistentRealization:
        if state == O2IState.LOS:
            return self.__los_realization
        elif state == O2IState.NLOS:
            return self.__nlos_realization
        else:
            return self.__o2i_realization

    def _sample_large_scale_state(
        self, state_variable_sample: float, parameters: ClusterDelayLineSampleParameters
    ) -> O2IState:
        los_probability = 18 / parameters.distance_3d + np.exp(-parameters.distance_3d / 36.0) * (
            1 - 18 / parameters.distance_3d
        )

        return O2IState.LOS if state_variable_sample < los_probability else O2IState.NLOS

    @staticmethod
    def __parameter_dependency(carrier_frequency: float, factor: float, summand: float) -> float:
        """An implementation of the frequently used equation

        .. math::

           y = a \\log_{10}(1 + f_c) + b

        Args:
            carrier_frequency: Carrier frequency
            factor: Factor scaling the logarithmic frequency dependency.
            summand: Added constant.

        Returns: The result.
        """

        fc = (
            max(2e9, carrier_frequency) * 1e-9
        )  # Frequency is lower-bounded by 2 GHz, according to Note 7 in table 7.5-6 of TR 138.901 v17.0.0
        return factor * log10(1 + fc) + summand

    # Parameters for computing the mean delay spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __delay_spread_mean: Mapping[O2IState, Tuple[float, float]] = {
        O2IState.LOS: (-0.24, -7.14),
        O2IState.NLOS: (-0.24, -6.83),
        O2IState.O2I: (0.0, -6.62),
    }

    @staticmethod
    def _delay_spread_mean(state: O2IState, carrier_frequency: float) -> float:
        mean_parameters = UrbanMicrocellsRealization.__delay_spread_mean[state]
        return UrbanMicrocellsRealization.__parameter_dependency(
            carrier_frequency, *mean_parameters
        )

    # Parameters for computing the standard deviation of the delay spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __delay_spread_std: Mapping[O2IState, Tuple[float, float]] = {
        O2IState.LOS: (0.0, 0.38),
        O2IState.NLOS: (0.16, 0.28),
        O2IState.O2I: (0.0, 0.32),
    }

    @staticmethod
    def _delay_spread_std(state: O2IState, carrier_frequency: float) -> float:
        std_parameters = UrbanMicrocellsRealization.__delay_spread_std[state]
        return UrbanMicrocellsRealization.__parameter_dependency(carrier_frequency, *std_parameters)

    # Parameters for computing the mean angle of departure spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aod_spread_mean: Mapping[O2IState, Tuple[float, float]] = {
        O2IState.LOS: (-0.05, 1.21),
        O2IState.NLOS: (-0.23, 1.53),
        O2IState.O2I: (0.0, 1.25),
    }

    @staticmethod
    def _aod_spread_mean(state: O2IState, carrier_frequency: float) -> float:
        mean_parameters = UrbanMicrocellsRealization.__aod_spread_mean[state]
        return UrbanMicrocellsRealization.__parameter_dependency(
            carrier_frequency, *mean_parameters
        )

    # Parameters for computing the standard deviation of the angle of departure spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aod_spread_std: Mapping[O2IState, Tuple[float, float]] = {
        O2IState.LOS: (0.0, 0.41),
        O2IState.NLOS: (0.11, 0.33),
        O2IState.O2I: (0.0, 0.42),
    }

    @staticmethod
    def _aod_spread_std(state: O2IState, carrier_frequency: float) -> float:
        std_parameters = UrbanMicrocellsRealization.__aod_spread_std[state]
        return UrbanMicrocellsRealization.__parameter_dependency(carrier_frequency, *std_parameters)

    # Parameters for computing the mean angle of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aoa_spread_mean: Mapping[O2IState, Tuple[float, float]] = {
        O2IState.LOS: (-0.08, 1.73),
        O2IState.NLOS: (-0.08, 1.81),
        O2IState.O2I: (0.0, 1.76),
    }

    @staticmethod
    def _aoa_spread_mean(state: O2IState, carrier_frequency: float) -> float:
        mean_parameters = UrbanMicrocellsRealization.__aoa_spread_mean[state]
        return UrbanMicrocellsRealization.__parameter_dependency(
            carrier_frequency, *mean_parameters
        )

    # Parameters for computing the standard deviation of the angle of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __aoa_spread_std: Mapping[O2IState, Tuple[float, float]] = {
        O2IState.LOS: (0.014, 0.28),
        O2IState.NLOS: (0.05, 0.3),
        O2IState.O2I: (0.0, 0.16),
    }

    @staticmethod
    def _aoa_spread_std(state: O2IState, carrier_frequency: float) -> float:
        std_parameters = UrbanMicrocellsRealization.__aoa_spread_std[state]
        return UrbanMicrocellsRealization.__parameter_dependency(carrier_frequency, *std_parameters)

    # Parameters for computing the mean zenith of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __zoa_spread_mean: Mapping[O2IState, Tuple[float, float]] = {
        O2IState.LOS: (-0.1, 0.73),
        O2IState.NLOS: (-0.04, 0.92),
        O2IState.O2I: (0.0, 1.01),
    }

    @staticmethod
    def _zoa_spread_mean(state: O2IState, carrier_frequency: float) -> float:
        mean_parameters = UrbanMicrocellsRealization.__zoa_spread_mean[state]
        return UrbanMicrocellsRealization.__parameter_dependency(
            carrier_frequency, *mean_parameters
        )

    # Parameters for computing the standard deviation of the zenith of arrival spread
    # TR 138.901 v17.0.0 Table 7.5-6
    __zoa_spread_std: Mapping[O2IState, Tuple[float, float]] = {
        O2IState.LOS: (-0.04, 0.34),
        O2IState.NLOS: (-0.07, 0.41),
        O2IState.O2I: (0.0, 0.43),
    }

    @staticmethod
    def _zoa_spread_std(state: O2IState, carrier_frequency: float) -> float:
        std_parameters = UrbanMicrocellsRealization.__zoa_spread_std[state]
        return UrbanMicrocellsRealization.__parameter_dependency(carrier_frequency, *std_parameters)

    @staticmethod
    def _rice_factor_mean() -> float:
        return 9.0

    @staticmethod
    def _rice_factor_std() -> float:
        return 5.0

    # Delay scaling factors for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __delay_scaling: Mapping[O2IState, float] = {
        O2IState.LOS: 3.0,
        O2IState.NLOS: 2.1,
        O2IState.O2I: 2.2,
    }

    @staticmethod
    def _delay_scaling(state: O2IState) -> float:
        return UrbanMicrocellsRealization.__delay_scaling[state]

    # Mean cross-polarization power ratio for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __cross_polarization_power_mean: Mapping[O2IState, float] = {
        O2IState.LOS: 9.0,
        O2IState.NLOS: 8.0,
        O2IState.O2I: 9.0,
    }

    @staticmethod
    def _cross_polarization_power_mean(state: O2IState) -> float:
        return UrbanMicrocellsRealization.__cross_polarization_power_mean[state]

    # Standard deviation of the cross-polarization power ratio for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __cross_polarization_power_std: Mapping[O2IState, float] = {
        O2IState.LOS: 3.0,
        O2IState.NLOS: 3.0,
        O2IState.O2I: 5.0,
    }

    @staticmethod
    def _cross_polarization_power_std(state: O2IState) -> float:
        return UrbanMicrocellsRealization.__cross_polarization_power_std[state]

    # Number of clusters for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    __num_clusters: Mapping[O2IState, int] = {O2IState.LOS: 12, O2IState.NLOS: 19, O2IState.O2I: 12}

    @staticmethod
    def _num_clusters(state: O2IState) -> int:
        return UrbanMicrocellsRealization.__num_clusters[state]

    # RMS cluster delay spread for different LOS states
    # TR 138.901 v17.0.0 Table 7.5-6
    # pragma: no cover
    __cluster_delay_spread: Mapping[O2IState, float] = {
        O2IState.LOS: 5.0 * 1e-9,
        O2IState.NLOS: 11.0 * 1e-9,
        O2IState.O2I: 11.0 * 1e-9,
    }

    @staticmethod
    def _cluster_delay_spread(state: O2IState, carrier_frequency: float) -> float:
        return UrbanMicrocellsRealization.__cluster_delay_spread[state]  # pragma: no cover

    # RMS cluster azimuth of departure spread for different LOS states in degrees
    # TR 138.901 v17.0.0 Table 7.5-6
    __cluster_aod_spread: Mapping[O2IState, float] = {
        O2IState.LOS: 3.0,
        O2IState.NLOS: 10.0,
        O2IState.O2I: 5.0,
    }

    @staticmethod
    def _cluster_aod_spread(state: O2IState) -> float:
        return UrbanMicrocellsRealization.__cluster_aod_spread[state]

    # RMS cluster azimuth of arrival spread for different LOS states in degrees
    # TR 138.901 v17.0.0 Table 7.5-6
    __cluster_aoa_spread: Mapping[O2IState, float] = {
        O2IState.LOS: 17.0,
        O2IState.NLOS: 22.0,
        O2IState.O2I: 8.0,
    }

    @staticmethod
    def _cluster_aoa_spread(state: O2IState) -> float:
        return UrbanMicrocellsRealization.__cluster_aoa_spread[state]

    # RMS cluster zenith of arrival spread for different LOS states in degrees
    # TR 138.901 v17.0.0 Table 7.5-6
    __cluster_zoa_spread: Mapping[O2IState, float] = {
        O2IState.LOS: 7.0,
        O2IState.NLOS: 7.0,
        O2IState.O2I: 3.0,
    }

    @staticmethod
    def _cluster_zoa_spread(state: O2IState) -> float:
        return UrbanMicrocellsRealization.__cluster_zoa_spread[state]

    # Standard deviation of the shadowing for different LOS states in dB
    # TR 138.901 v17.0.0 Table 7.5-6
    __cluster_shadowing_std: Mapping[O2IState, float] = {
        O2IState.LOS: 3.0,
        O2IState.NLOS: 3.0,
        O2IState.O2I: 4.0,
    }

    @staticmethod
    def _cluster_shadowing_std(state: O2IState) -> float:
        return UrbanMicrocellsRealization.__cluster_shadowing_std[state]

    @staticmethod
    def _zod_spread_mean(state: O2IState, parameters: ClusterDelayLineSampleParameters) -> float:
        # Implementation of TR 138.901 v17.0.0 Table 7.5-8
        if state == O2IState.LOS:
            return (
                max(-0.21, -14.8 * parameters.distance_2d / 1000)
                + 0.01 * abs(parameters.terminal_height - parameters.base_height)
                + 0.83
            )
        else:
            return (
                max(-0.5, -3.1 * parameters.distance_2d / 1000)
                + 0.01 * max(parameters.terminal_height - parameters.base_height, 0.0)
                + 0.2
            )

    @staticmethod
    def _zod_spread_std(state: O2IState, parameters: ClusterDelayLineSampleParameters) -> float:
        # TR 138.901 v17.0.0 Table 7.5-8
        return 0.35

    @staticmethod
    def _zod_offset(state: O2IState, parameters: ClusterDelayLineSampleParameters) -> float:
        if state == O2IState.LOS:
            return 0.0
        else:
            return -(10 ** (-1.5 * log10(max(10, parameters.terminal_height)) + 3.3))

    @override
    def serialize(self, process: SerializationProcess) -> None:
        ClusterDelayLineRealization.serialize(self, process)
        process.serialize_object(self.__los_realization, "los_realization")
        process.serialize_object(self.__nlos_realization, "nlos_realization")
        process.serialize_object(self.__o2i_realization, "o2i_realization")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> UrbanMicrocellsRealization:
        return UrbanMicrocellsRealization(
            expected_state=process.deserialize_object("expected_state", O2IState, None),
            los_realization=process.deserialize_object("los_realization", ConsistentRealization),
            nlos_realization=process.deserialize_object("nlos_realization", ConsistentRealization),
            o2i_realization=process.deserialize_object("o2i_realization", ConsistentRealization),
            sample_hooks=set(),
            **ClusterDelayLineRealization._DeserializeParameters(process),  # type: ignore[arg-type]
        )


class UrbanMicrocells(ClusterDelayLineBase[UrbanMicrocellsRealization, O2IState]):
    """3GPP cluster delay line preset modeling an urban street canyon."""

    @property
    def max_num_clusters(self) -> int:
        return 19

    @property
    def max_num_rays(self) -> int:
        return 20

    @property
    def _large_scale_correlations(self) -> np.ndarray:
        # Large scale cross correlations
        # TR 138.901 v17.0.0 Table 7.5-6
        return np.array(
            [
                #    LOS   NLOS  O2I
                [+0.5, +0.0, +0.4],  # 0: ASD vs DS
                [+0.8, +0.4, +0.4],  # 1: ASA vs DS
                [-0.4, -0.4, +0.0],  # 2: ASA VS SF
                [-0.5, +0.0, +0.2],  # 3: ASD vs SF
                [-0.4, -0.7, -0.5],  # 4: DS vs SF
                [+0.4, +0.0, +0.0],  # 5: ASD vs ASA
                [-0.2, +0.0, +0.0],  # 6: ASD vs K
                [-0.3, +0.0, +0.0],  # 7: ASA vs K
                [-0.7, +0.0, +0.0],  # 8: DS vs K
                [+0.5, +0.0, +0.0],  # 9: SF vs K
                [+0.0, +0.0, +0.0],  # 10: ZSD vs SF
                [+0.0, +0.0, +0.0],  # 11: ZSA vs SF
                [+0.0, +0.0, +0.0],  # 12: ZSD vs K
                [+0.0, +0.0, +0.0],  # 13: ZSA vs K
                [+0.0, -0.5, -0.6],  # 14: ZSD vs DS
                [+0.2, +0.0, -0.2],  # 15: ZSA vs DS
                [+0.5, +0.5, -0.2],  # 16: ZSD vs ASD
                [+0.3, +0.5, +0.0],  # 17: ZSA vs ASD
                [+0.0, +0.0, +0.0],  # 18: ZSD vs ASA
                [+0.0, +0.2, +0.5],  # 19: ZSA vs ASA
                [+0.0, +0.0, +0.5],  # 20: ZSD vs ZSA
            ],
            dtype=np.float64,
        ).T

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> UrbanMicrocells:
        return UrbanMicrocells(**ClusterDelayLineBase._DeserializeParameters(process))  # type: ignore[arg-type]

    def _initialize_realization(
        self,
        state_generator: ConsistentGenerator,
        parameter_generator: ConsistentGenerator,
        parameters: ClusterDelayLineRealizationParameters,
    ) -> UrbanMicrocellsRealization:

        # Generate realizations for each large scale state
        # TR 138.901 v17.0.0 Table 7.6.3.1-2
        state_realization = state_generator.realize(50.0)
        los_realization = parameter_generator.realize(12.0)
        nlos_realization = parameter_generator.realize(15.0)
        o2i_realization = parameter_generator.realize(15.0)

        return UrbanMicrocellsRealization(
            self.expected_state,
            state_realization,
            los_realization,
            nlos_realization,
            o2i_realization,
            parameters,
            self.sample_hooks,
            self.gain,
        )
