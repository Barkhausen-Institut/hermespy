# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

import numpy as np

from hermespy.core import DeserializationProcess, SerializableEnum, SerializationProcess
from .fading import AntennaCorrelation, MultipathFadingChannel
from ..channel import Channel

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Cost259Type(SerializableEnum):
    """Supported model types of the Cost256 channel model"""

    URBAN = 0
    """Urban area"""

    RURAL = 1
    """Rural area"""

    HILLY = 2
    """Hilly terrain"""


class Cost259(MultipathFadingChannel):
    """Cost action 259 multipath fading channel model."""

    __DEFAULT_TYPE = Cost259Type.URBAN

    __model_type: Cost259Type

    def __init__(
        self,
        model_type: Cost259Type = __DEFAULT_TYPE,
        correlation_distance: float = MultipathFadingChannel._DEFAULT_DECORRELATION_DISTANCE,
        num_sinusoids: int = MultipathFadingChannel._DEFAULT_NUM_SINUSOIDS,
        los_angle: float | None = None,
        doppler_frequency: float = MultipathFadingChannel._DEFAULT_DOPPLER_FREQUENCY,
        los_doppler_frequency: float | None = None,
        antenna_correlation: AntennaCorrelation | None = None,
        gain: float = MultipathFadingChannel._DEFAULT_GAIN,
        seed: int | None = None,
    ) -> None:
        """
        Args:

            model_type (Cost259Type):
                The model type.
                By default, an urban model is assumed.

            correlation_distance (float, optional):
                Distance at which channel samples are considered to be uncorrelated.
                :math:`\\infty` by default, i.e. the channel is considered to be fully correlated in space.

            num_sinusoids (int, optional):
                Number of sinusoids used to sample the statistical distribution.
                Denoted by :math:`N` within the respective equations.

            los_angle (float, optional):
                Angle phase of the line of sight component within the statistical distribution.
                Will be ignored for the Hilly model type.

            doppler_frequency (float, optional):
                Doppler frequency shift of the statistical distribution.
                Denoted by :math:`\\omega_{\\ell}` within the respective equations.

            antenna_correlation (AntennaCorrelation, optional):
                Antenna correlation model.
                By default, the channel assumes ideal correlation, i.e. no cross correlations.

            gain (float, optional):
                Linear power gain factor a signal experiences when being propagated over this realization.
                :math:`1.0` by default.

            seed (int, optional):
                Seed used to initialize the pseudo-random number generator.

        Raises:
           ValueError:
                If `model_type` is not supported.
                If `los_angle` is defined in HILLY model type.
        """

        if model_type == Cost259Type.URBAN:
            delays = 1e-6 * np.array(
                [
                    0,
                    0.217,
                    0.512,
                    0.514,
                    0.517,
                    0.674,
                    0.882,
                    1.230,
                    1.287,
                    1.311,
                    1.349,
                    1.533,
                    1.535,
                    1.622,
                    1.818,
                    1.836,
                    1.884,
                    1.943,
                    2.048,
                    2.140,
                ]
            )
            power_db = np.array(
                [
                    -5.7,
                    -7.6,
                    -10.1,
                    -10.2,
                    -10.2,
                    -11.5,
                    -13.4,
                    -16.3,
                    -16.9,
                    -17.1,
                    -17.4,
                    -19.0,
                    -19.0,
                    -19.8,
                    -21.5,
                    -21.6,
                    -22.1,
                    -22.6,
                    -23.5,
                    -24.3,
                ]
            )
            rice_factors = np.zeros(delays.shape)

        elif model_type == Cost259Type.RURAL:
            delays = 1e-6 * np.array(
                [0, 0.042, 0.101, 0.129, 0.149, 0.245, 0.312, 0.410, 0.469, 0.528]
            )
            power_db = np.array([-5.2, -6.4, -8.4, -9.3, -10.0, -13.1, -15.3, -18.5, -20.4, -22.4])
            rice_factors = np.zeros(delays.shape)

        elif model_type == Cost259Type.HILLY:
            delays = 1e-6 * np.array(
                [
                    0,
                    0.356,
                    0.441,
                    0.528,
                    0.546,
                    0.609,
                    0.625,
                    0.842,
                    0.916,
                    0.941,
                    15.0,
                    16.172,
                    16.492,
                    16.876,
                    16.882,
                    16.978,
                    17.615,
                    17.827,
                    17.849,
                    18.016,
                ]
            )
            power_db = np.array(
                [
                    -3.6,
                    -8.9,
                    -10.2,
                    -11.5,
                    -11.8,
                    -12.7,
                    -13.0,
                    -16.2,
                    -17.3,
                    -17.7,
                    -17.6,
                    -22.7,
                    -24.1,
                    -25.8,
                    -25.8,
                    -26.2,
                    -29.0,
                    -29.9,
                    -30.0,
                    -30.7,
                ]
            )
            rice_factors = np.hstack([np.array([np.inf]), np.zeros(delays.size - 1)])
            los_angle = np.arccos(0.7)

        else:
            raise ValueError("Requested model type not supported")

        self.__model_type = Cost259Type(model_type)

        # Convert power and normalize
        power_profile = 10 ** (power_db / 10)
        power_profile /= sum(power_profile)

        # Init base class with pre-defined model parameters
        MultipathFadingChannel.__init__(
            self,
            delays,
            power_profile,
            rice_factors,
            correlation_distance,
            num_sinusoids,
            los_angle,
            doppler_frequency,
            los_doppler_frequency,
            antenna_correlation,
            gain,
            seed,
        )

    @property
    def model_type(self) -> Cost259Type:
        """Access the configured model type.

        Returns: The configured model type.
        """

        return self.__model_type

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object(self.model_type, "model_type")
        process.serialize_floating(self.correlation_distance, "correlation_distance")
        process.serialize_integer(self.num_sinusoids, "num_sinusoids")
        process.serialize_floating(self.los_angle, "los_angle")
        process.serialize_floating(self.doppler_frequency, "doppler_frequency")
        process.serialize_floating(self.los_doppler_frequency, "los_doppler_frequency")
        if self.antenna_correlation is not None:
            process.serialize_object(self.antenna_correlation, "antenna_correlation")
        Channel.serialize(self, process)

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> Cost259:
        return cls(
            model_type=process.deserialize_object("model_type", Cost259Type, cls.__DEFAULT_TYPE),
            correlation_distance=process.deserialize_floating(
                "correlation_distance", cls._DEFAULT_DECORRELATION_DISTANCE
            ),
            num_sinusoids=process.deserialize_integer("num_sinusoids", cls._DEFAULT_NUM_SINUSOIDS),
            los_angle=process.deserialize_floating("los_angle"),
            doppler_frequency=process.deserialize_floating(
                "doppler_frequency", cls._DEFAULT_DOPPLER_FREQUENCY
            ),
            los_doppler_frequency=process.deserialize_floating("los_doppler_frequency"),
            antenna_correlation=process.deserialize_object(
                "antenna_correlation", AntennaCorrelation, None
            ),
            **Channel._DeserializeParameters(process),  # type: ignore[arg-type]
        )
