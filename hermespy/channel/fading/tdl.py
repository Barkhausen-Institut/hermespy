# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

from hermespy.core import SerializableEnum
from .fading import MultipathFadingChannel

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TDLType(SerializableEnum):
    """Supported model types of the 5G TDL channel model"""

    A = 0
    B = 1
    C = 2
    D = 4
    E = 5


class TDL(MultipathFadingChannel):
    """5G TDL Multipath Fading Channel models."""

    yaml_tag = "5GTDL"
    __rms_delay: float

    def __init__(
        self,
        model_type: TDLType = TDLType.A,
        rms_delay: float = 0.0,
        gain: float = 1.0,
        doppler_frequency: float | None = None,
        los_doppler_frequency: float | None = None,
        **kwargs,
    ) -> None:
        """
        Args:

            model_type (TYPE):
                The model type.
                Initializes the :attr:`model_type` attribute.

            rms_delay (float):
                Root-Mean-Squared delay in seconds.
                Initializes the :attr:`rms_delay` attribute.

            alpha_device (SimulatedDevice, optional):
                First device linked by this :class:`MultipathFading5GTDL` channel instance.
                Initializes the :attr:`alpha_device` property.
                If not specified the channel is considered floating,
                meaning a call to :meth:`realize<Channel.realize>` will raise an exception.

            beta_device (SimulatedDevice, optional):
                Second device linked by this :class:`MultipathFading5GTDL` channel.
                Initializes the :attr:`beta_device` property.
                If not specified the channel is considered floating,
                meaning a call to :meth:`realize` will raise an exception.

            num_sinusoids (int, optional):
                Number of sinusoids used to sample the statistical distribution.

            doppler_frequency (float, optional)
                Doppler frequency shift of the statistical distribution.

            \***kwargs (Any):
                Additional `MultipathFadingChannel` initialization parameters.

        Raises:

            ValueError: If `rms_delay` is smaller than zero.
            ValueError: If `los_angle` is specified in combination with `model_type` D or E.
        """

        if rms_delay < 0.0:
            raise ValueError("Root-Mean-Squared delay must be greater or equal to zero")

        self.__rms_delay = rms_delay

        # ETSI TR 38.900 Table 7.7.2-1: 5G TDL-A
        if model_type == TDLType.A:
            normalized_delays = np.array(
                [
                    0,
                    0.3819,
                    0.4025,
                    0.5868,
                    0.4610,
                    0.5375,
                    0.6708,
                    0.5750,
                    0.7618,
                    1.5375,
                    1.8978,
                    2.2242,
                    2.1717,
                    2.4942,
                    2.5119,
                    3.0582,
                    4.0810,
                    4.4579,
                    4.5695,
                    4.7966,
                    5.0066,
                    5.3043,
                    9.6586,
                ]
            )
            power_db = np.array(
                [
                    -13.4,
                    0,
                    -2.2,
                    -4,
                    -6,
                    -8.2,
                    -9.9,
                    -10.5,
                    -7.5,
                    -15.9,
                    -6.6,
                    -16.7,
                    -12.4,
                    -15.2,
                    -10.8,
                    -11.3,
                    -12.7,
                    -16.2,
                    -18.3,
                    -18.9,
                    -16.6,
                    -19.9,
                    -29.7,
                ]
            )
            rice_factors = np.zeros(normalized_delays.shape)

        # ETSI TR 38.900 Table 7.7.2-2: 5G TDL-B
        elif model_type == TDLType.B:
            normalized_delays = np.array(
                [
                    0,
                    0.1072,
                    0.2155,
                    0.2095,
                    0.2870,
                    0.2986,
                    0.3752,
                    0.5055,
                    0.3681,
                    0.3697,
                    0.5700,
                    0.5283,
                    1.1021,
                    1.2756,
                    1.5474,
                    1.7842,
                    2.0169,
                    2.8294,
                    3.0219,
                    3.6187,
                    4.1067,
                    4.2790,
                    4.7834,
                ]
            )

            # ETSI TR 38.900 Table 7.7.2-3: 5G TDL-C
            power_db = np.array(
                [
                    0,
                    -2.2,
                    -4,
                    -3.2,
                    -9.8,
                    -3.2,
                    -3.4,
                    -5.2,
                    -7.6,
                    -3,
                    -8.9,
                    -9,
                    -4.8,
                    -5.7,
                    -7.5,
                    -1.9,
                    -7.6,
                    -12.2,
                    -9.8,
                    -11.4,
                    -14.9,
                    -9.2,
                    -11.3,
                ]
            )
            rice_factors = np.zeros(normalized_delays.shape)

        # ETSI TR 38.900 Table 7.7.2-3: 5G TDL-C
        elif model_type == TDLType.C:
            normalized_delays = np.array(
                [
                    0,
                    0.2099,
                    0.2219,
                    0.2329,
                    0.2176,
                    0.6366,
                    0.6448,
                    0.6560,
                    0.6584,
                    0.7935,
                    0.8213,
                    0.9336,
                    1.2285,
                    1.3083,
                    2.1704,
                    2.7105,
                    4.2589,
                    4.6003,
                    5.4902,
                    5.6077,
                    6.3065,
                    6.6374,
                    7.0427,
                    8.6523,
                ]
            )
            power_db = np.array(
                [
                    -4.4,
                    -1.2,
                    -3.5,
                    -5.2,
                    -2.5,
                    0,
                    -2.2,
                    -3.9,
                    -7.4,
                    -7.1,
                    -10.7,
                    -11.1,
                    -5.1,
                    -6.8,
                    -8.7,
                    -13.2,
                    -13.9,
                    -13.9,
                    -15.8,
                    -17.1,
                    -16,
                    -15.7,
                    -21.6,
                    -22.8,
                ]
            )
            rice_factors = np.zeros(normalized_delays.shape)

        # ETSI TR 38.900 Table 7.7.2-4: 5G TDL-D
        elif model_type == TDLType.D:
            if los_doppler_frequency is not None:
                raise ValueError(
                    "Model type D does not support line of sight doppler frequency configuration"
                )

            normalized_delays = np.array(
                [
                    0,
                    0.035,
                    0.612,
                    1.363,
                    1.405,
                    1.804,
                    2.596,
                    1.775,
                    4.042,
                    7.937,
                    9.424,
                    9.708,
                    12.525,
                ]
            )
            power_db = np.array(
                [
                    -13.5,
                    -18.8,
                    -21,
                    -22.8,
                    -17.9,
                    -20.1,
                    -21.9,
                    -22.9,
                    -27.8,
                    -23.6,
                    -24.8,
                    -30.0,
                    -27.7,
                ]
            )
            rice_factors = np.zeros(normalized_delays.shape)
            rice_factors[0] = 13.3
            los_doppler_frequency = 0.7

        # ETSI TR 38.900 Table 7.7.2-5: 5G TDL-E
        elif model_type == TDLType.E:
            if los_doppler_frequency is not None:
                raise ValueError(
                    "Model type E does not support line of sight doppler frequency configuration"
                )

            normalized_delays = np.array(
                [
                    0,
                    0.5133,
                    0.5440,
                    0.5630,
                    0.5440,
                    0.7112,
                    1.9092,
                    1.9293,
                    1.9589,
                    2.6426,
                    3.7136,
                    5.4524,
                    12.0034,
                    20.6519,
                ]
            )
            power_db = np.array(
                [
                    -22.03,
                    -15.8,
                    -18.1,
                    -19.8,
                    -22.9,
                    -22.4,
                    -18.6,
                    -20.8,
                    -22.6,
                    -22.3,
                    -25.6,
                    -20.2,
                    -29.8,
                    -29.2,
                ]
            )
            rice_factors = np.zeros(normalized_delays.shape)
            rice_factors[0] = 22
            los_doppler_frequency = 0.7

        else:
            raise ValueError("Requested model type not supported")

        self.__model_type = TDLType(model_type)

        # Convert power and normalize
        power_profile = 10 ** (power_db / 10)
        power_profile /= sum(power_profile)

        # Scale delays
        delays = rms_delay * normalized_delays

        # Init base class with pre-defined model parameters
        MultipathFadingChannel.__init__(
            self,
            gain=gain,
            delays=delays,
            power_profile=power_profile,
            rice_factors=rice_factors,
            doppler_frequency=doppler_frequency,
            los_doppler_frequency=los_doppler_frequency,
            **kwargs,
        )

    @property
    def model_type(self) -> TDLType:
        """Access the configured model type.

        Returns:
            MultipathFading5gTDL.TYPE: The configured model type.
        """

        return self.__model_type

    @property
    def rms_delay(self) -> float:
        """Root mean squared channel delay.

        Returns: Delay in seconds.
        """

        return self.__rms_delay
