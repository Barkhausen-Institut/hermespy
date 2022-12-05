# -*- coding: utf-8 -*-
"""
===================================
Multipath Fading Standard Templates
===================================
"""

from __future__ import annotations
from typing import Any, Optional, Type, Union

import numpy as np
from ruamel.yaml import SafeRepresenter, MappingNode

from hermespy.core import FloatingError, Serializable, SerializableEnum
from .multipath_fading_channel import AntennaCorrelation, MultipathFadingChannel

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DeviceType(SerializableEnum):
    """3GPP device type"""

    BASE_STATION = 0
    """Base station"""

    TERMINAL = 1
    """Mobile terminal"""


class CorrelationType(SerializableEnum):
    """3GPP correlation type"""

    LOW = 0.0, 0.0
    """Low antenna correlation"""

    MEDIUM = 0.3, 0.3
    """Medium antenna correlation"""

    MEDIUM_A = 0.3, 0.3874
    """Medium antenna correlation"""

    HIGH = 0.9, 0.9
    """High antenna correlation"""


class StandardAntennaCorrelation(Serializable, AntennaCorrelation):
    """3GPP 5G Multipath fading standardized antenna correlations"""

    yaml_tag = "StandardCorrelation"
    """YAML serialization tag"""

    __device_type: DeviceType  # The assumed device
    __correlation: CorrelationType  # The assumed correlation

    def __init__(self, device_type: Union[DeviceType, int, str], correlation: Union[CorrelationType, str], **kwargs) -> None:
        """
        Args:

            device_type (Union[DeviceType, int, str]):
                The assumed device.

            correlation (Union[CorrelationType, str]):
                The assumed correlation.
        """

        self.device_type = device_type
        self.correlation = correlation

        AntennaCorrelation.__init__(self, **kwargs)

    @property
    def device_type(self) -> DeviceType:
        """Assumed 3GPP device type.

        Returns: The device type.

        Raises:

            ValuError: On unsupported type conversions.
        """

        return self.__device_type

    @device_type.setter
    def device_type(self, value: Union[DeviceType, int, str]) -> None:

        if isinstance(value, DeviceType):
            self.__device_type = value

        elif isinstance(value, int):
            self.__device_type = DeviceType(value)

        elif isinstance(value, str):
            self.__device_type = DeviceType[value]

        else:
            raise ValueError("Unknown device_type type")

    @property
    def correlation(self) -> CorrelationType:
        """Assumed 3GPP standard correlation type.

        Returns: The correlation type.

        Raises:

            ValuError: On unsupported type conversions.
        """

        return self.__correlation

    @correlation.setter
    def correlation(self, value: Union[CorrelationType, str]) -> None:

        if isinstance(value, CorrelationType):
            self.__correlation = value

        elif isinstance(value, str):
            self.__correlation = CorrelationType[value]

        else:
            raise ValueError("Unsupported correlation type conversion")

    @property
    def covariance(self) -> np.ndarray:

        if self.device is None:
            raise FloatingError("Error trying to compute the covariance matrix of an unknown device")

        f = self.__correlation.value[self.__device_type.value]
        n = self.device.num_antennas

        if n == 1:
            return np.ones((1, 1), dtype=complex)

        if n == 2:
            return np.array([[1, f], [f, 1]], dtype=complex)

        if n == 4:
            return np.array([[1, f ** (1 / 9), f ** (4 / 9), f], [f ** (1 / 9), 1, f ** (1 / 9), f ** (4 / 9)], [f ** (4 / 9), f ** (1 / 9), 1, f ** (1 / 9)], [f, f ** (4 / 9), f ** (1 / 9), 1]], dtype=complex)

        raise RuntimeError(f"3GPP standard antenna covariance is only defined for 1, 2 and 4 antennas, device has {n} antennas")


class Cost256Type(SerializableEnum):
    """Supported model types of the Cost256 channel model"""

    URBAN = 0
    """Urban area"""

    RURAL = 1
    """Rural area"""

    HILLY = 2
    """Hilly terrain"""


class MultipathFadingCost256(MultipathFadingChannel):
    """COST256 Multipath Fading Channel models."""

    yaml_tag = "COST256"
    __model_type: Cost256Type

    def __init__(self, model_type: Cost256Type = Cost256Type.URBAN, los_angle: Optional[float] = None, doppler_frequency: Optional[float] = None, los_doppler_frequency: Optional[float] = None, **kwargs: Any) -> None:
        """Model initialization.

        Args:

            model_type (Cost256Type): The model type.

            los_angle (float, optional):
                Angle phase of the line of sight component within the statistical distribution.

            doppler_frequency (float, optional):
                Doppler frequency shift of the statistical distribution.

            kwargs (Any):
                `MultipathFadingChannel` initialization parameters.

        Raises:
           ValueError:
                If `model_type` is not supported.
                If `los_angle` is defined in HILLY model type.
        """

        if model_type == Cost256Type.URBAN:

            delays = 1e-6 * np.array([0, 0.217, 0.512, 0.514, 0.517, 0.674, 0.882, 1.230, 1.287, 1.311, 1.349, 1.533, 1.535, 1.622, 1.818, 1.836, 1.884, 1.943, 2.048, 2.140])
            power_db = np.array([-5.7, -7.6, -10.1, -10.2, -10.2, -11.5, -13.4, -16.3, -16.9, -17.1, -17.4, -19.0, -19.0, -19.8, -21.5, -21.6, -22.1, -22.6, -23.5, -24.3])
            rice_factors = np.zeros(delays.shape)

        elif model_type == Cost256Type.RURAL:

            delays = 1e-6 * np.array([0, 0.042, 0.101, 0.129, 0.149, 0.245, 0.312, 0.410, 0.469, 0.528])
            power_db = np.array([-5.2, -6.4, -8.4, -9.3, -10.0, -13.1, -15.3, -18.5, -20.4, -22.4])
            rice_factors = np.zeros(delays.shape)

        elif model_type == Cost256Type.HILLY:

            if los_angle is not None:
                raise ValueError("Model type HILLY does not support line of sight angle configuration")

            delays = 1e-6 * np.array([0, 0.356, 0.441, 0.528, 0.546, 0.609, 0.625, 0.842, 0.916, 0.941, 15.0, 16.172, 16.492, 16.876, 16.882, 16.978, 17.615, 17.827, 17.849, 18.016])
            power_db = np.array([-3.6, -8.9, -10.2, -11.5, -11.8, -12.7, -13.0, -16.2, -17.3, -17.7, -17.6, -22.7, -24.1, -25.8, -25.8, -26.2, -29.0, -29.9, -30.0, -30.7])
            rice_factors = np.hstack([np.array([np.inf]), np.zeros(delays.size - 1)])
            los_angle = np.arccos(0.7)

        else:
            raise ValueError("Requested model type not supported")

        self.__model_type = Cost256Type(model_type)

        # Convert power and normalize
        power_profile = 10 ** (power_db / 10)
        power_profile /= sum(power_profile)

        # Init base class with pre-defined model parameters
        MultipathFadingChannel.__init__(self, delays=delays, power_profile=power_profile, rice_factors=rice_factors, los_angle=los_angle, doppler_frequency=doppler_frequency, los_doppler_frequency=los_doppler_frequency, **kwargs)

    @property
    def model_type(self) -> Cost256Type:
        """Access the configured model type.

        Returns: The configured model type.
        """

        return self.__model_type

    @classmethod
    def to_yaml(cls: Type[MultipathFadingCost256], representer: SafeRepresenter, node: MultipathFadingCost256) -> MappingNode:
        """Serialize a serializable object to YAML.

        Args:

            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Serializable):
                The MultipathFadingCost256 instance to be serialized.

        Returns: The serialized YAML node.
        """

        blacklist = set()

        if node.model_type == Cost256Type.HILLY:
            blacklist.add("los_angle")

        return node._mapping_serialization_wrapper(representer, blacklist=blacklist)


class TDLType(SerializableEnum):
    """Supported model types of the 5G TDL channel model"""

    A = 0
    B = 1
    C = 2
    D = 4
    E = 5


class MultipathFading5GTDL(MultipathFadingChannel):
    """5G TDL Multipath Fading Channel models."""

    yaml_tag = "5GTDL"
    __rms_delay: float

    def __init__(self, model_type: TDLType = TDLType.A, rms_delay: float = 0.0, doppler_frequency: Optional[float] = None, los_doppler_frequency: Optional[float] = None, **kwargs: Any) -> None:
        """Model initialization.

        Args:

            model_type (TYPE): The model type.
            rms_delay (float): Root-Mean-Squared delay in seconds.

            num_sinusoids (int, optional):
                Number of sinusoids used to sample the statistical distribution.

            doppler_frequency (float, optional)
                Doppler frequency shift of the statistical distribution.

            kwargs (Any):
                `MultipathFadingChannel` initialization parameters.

        Raises:
            ValueError:
                If `model_type` is not supported.
                If `rms_delay` is smaller than zero.
                If `los_angle` is specified in combination with `model_type` D or E.
        """

        if rms_delay < 0.0:
            raise ValueError("Root-Mean-Squared delay must be greater or equal to zero")

        self.__rms_delay = rms_delay

        if model_type == TDLType.A:

            normalized_delays = np.array([0, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750, 0.7618, 1.5375, 1.8978, 2.2242, 2.1717, 2.4942, 2.5119, 3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586])
            power_db = np.array([-13.4, 0, -2.2, -4, -6, -8.2, -9.9, -10.5, -7.5, -15.9, -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2, -18.3, -18.9, -16.6, -19.9, -29.7])
            rice_factors = np.zeros(normalized_delays.shape)

        elif model_type == TDLType.B:

            normalized_delays = np.array([0, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752, 0.5055, 0.3681, 0.3697, 0.5700, 0.5283, 1.1021, 1.2756, 1.5474, 1.7842, 2.0169, 2.8294, 3.0219, 3.6187, 4.1067, 4.2790, 4.7834])
            power_db = np.array([0, -2.2, -4, -3.2, -9.8, -3.2, -3.4, -5.2, -7.6, -3, -8.9, -9, -4.8, -5.7, -7.5, -1.9, -7.6, -12.2, -9.8, -11.4, -14.9, -9.2, -11.3])
            rice_factors = np.zeros(normalized_delays.shape)

        elif model_type == TDLType.C:

            normalized_delays = np.array([0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560, 0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083, 2.1704, 2.7105, 4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523])
            power_db = np.array([-4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9, -7.4, -7.1, -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, -13.9, -15.8, -17.1, -16, -15.7, -21.6, -22.8])
            rice_factors = np.zeros(normalized_delays.shape)

        elif model_type == TDLType.D:

            if los_doppler_frequency is not None:
                raise ValueError("Model type D does not support line of sight doppler frequency configuration")

            normalized_delays = np.array([0, 0.035, 0.612, 1.363, 1.405, 1.804, 2.596, 1.775, 4.042, 7.937, 9.424, 9.708, 12.525])
            power_db = np.array([-13.5, -18.8, -21, -22.8, -17.9, -20.1, -21.9, -22.9, -27.8, -23.6, -24.8, -30.0, -27.7])
            rice_factors = np.zeros(normalized_delays.shape)
            rice_factors[0] = 13.3
            los_doppler_frequency = 0.7

        elif model_type == TDLType.E:

            if los_doppler_frequency is not None:
                raise ValueError("Model type E does not support line of sight doppler frequency configuration")

            normalized_delays = np.array([0, 0.5133, 0.5440, 0.5630, 0.5440, 0.7112, 1.9092, 1.9293, 1.9589, 2.6426, 3.7136, 5.4524, 12.0034, 20.6519])
            power_db = np.array([-22.03, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6, -20.8, -22.6, -22.3, -25.6, -20.2, -29.8, -29.2])
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
        MultipathFadingChannel.__init__(self, delays=delays, power_profile=power_profile, rice_factors=rice_factors, doppler_frequency=doppler_frequency, los_doppler_frequency=los_doppler_frequency, **kwargs)

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


class MultipathFadingExponential(MultipathFadingChannel):
    """Exponential Multipath Fading Channel models."""

    yaml_tag = "Exponential"

    __exponential_truncation: float = 1e-5
    __tap_interval: float
    __rms_delay: float

    def __init__(self, tap_interval: float = 0.0, rms_delay: float = 0.0, **kwargs: Any) -> None:
        """Exponential Multipath Channel Model initialization.

        Args:

            tap_interval (float, optional):
                Tap interval in seconds.

            rms_delay (float, optional):
                Root-Mean-Squared delay in seconds.

            kwargs (Any):
                `MultipathFadingChannel` initialization parameters.

        Raises:
            ValueError: On invalid arguments.
        """

        if tap_interval <= 0.0:
            raise ValueError("Tap interval must be greater than zero")

        if rms_delay <= 0.0:
            raise ValueError("Root-Mean-Squared delay must be greater than zero")

        self.__tap_interval = tap_interval
        self.__rms_delay = rms_delay

        rms_norm = rms_delay / tap_interval

        # Calculate the decay exponent alpha based on an infinite power delay profile, in which case
        # rms_delay = exp(-alpha/2)/(1-exp(-alpha)), cf. geometric distribution.
        # Truncate the distributions for paths whose average power is very
        # small (less than exponential_truncation).

        alpha = -2 * np.log((-1 + np.sqrt(1 + 4 * rms_norm**2)) / (2 * rms_norm))
        max_delay_in_samples = int(np.ceil(np.log(MultipathFadingExponential.__exponential_truncation) / alpha))

        delays = np.arange(max_delay_in_samples + 1) * tap_interval
        power_profile = np.exp(-alpha * np.arange(max_delay_in_samples + 1))
        rice_factors = np.zeros(delays.shape)

        # Init base class with pre-defined model parameters
        MultipathFadingChannel.__init__(self, delays=delays, power_profile=power_profile, rice_factors=rice_factors, **kwargs)

    @property
    def tap_interval(self) -> float:
        """Tap interval.

        Returns: Tap interval in seconds.
        """

        return self.__tap_interval

    @property
    def rms_delay(self) -> float:
        """Root mean squared channel delay.

        Returns: Delay in seconds.
        """

        return self.__rms_delay
