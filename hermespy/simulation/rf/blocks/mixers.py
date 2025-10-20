# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

from hermespy.core import SerializableEnum, SerializationProcess, DeserializationProcess
from ..block import RFBlock, RFBlockPort, RFBlockPortType, RFBlockRealization
from ..signal import RFSignal
from ..noise import NoiseModel, NoiseLevel

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MixerType(SerializableEnum):
    """Type of the considered mixer."""

    UP = 0
    """Up-converting mixer, converting the input signal to a higher frequency."""

    DOWN = 1
    """Down-converting mixer, converting the input signal to a lower frequency."""


class MixerRealization(RFBlockRealization):
    """Realization of a radio-frequency chain mixer block model."""

    __type: MixerType

    def __init__(
        self,
        sampling_rate: float,
        oversampling_factor: int,
        noise_realization,
        mixer_type: MixerType,
    ) -> None:
        """
        Args:
            sampling_rate: Sampling rate of the block in Hz.
            oversampling_factor: Oversampling factor of the modeling.
            noise_realization: Thermal noise realization applied to the mixer.
            mixer_type: Type of the mixer, either UP or DOWN.
        """

        # Initialize base class
        RFBlockRealization.__init__(self, sampling_rate, oversampling_factor, noise_realization)

        # Store attributes
        self.__type = mixer_type

    @property
    def mixer_type(self) -> MixerType:
        """Type of the mixer."""

        return self.__type


class Mixer(RFBlock):
    """Customizable three-port mixer block model."""

    __type: MixerType
    __i: RFBlockPort
    __o: RFBlockPort
    __lo: RFBlockPort

    def __init__(
        self,
        mixer_type: MixerType,
        noise_model: NoiseModel | None = None,
        noise_level: NoiseLevel | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            mixer_type: Type of the mixer, either UP or DOWN.
            seed: Seed with which to initialize the block's random state.
        """

        # Initialize base class
        RFBlock.__init__(self, noise_model, noise_level, seed)

        # Initialize class attributes
        self.__type = mixer_type
        self.__i = RFBlockPort(self, 0, RFBlockPortType.IN)
        self.__o = RFBlockPort(self, 0, RFBlockPortType.OUT)
        self.__lo = RFBlockPort(self, 1, RFBlockPortType.IN)

    @property
    def mixer_type(self) -> MixerType:
        """Type of the mixer."""

        return self.__type

    @property
    @override
    def num_input_ports(self) -> int:
        return 2

    @property
    @override
    def num_output_ports(self) -> int:
        return 1

    @override
    def realize(
        self, bandwidth: float, oversampling_factor: int, carrier_frequency: float
    ) -> MixerRealization:
        return MixerRealization(
            bandwidth,
            oversampling_factor,
            self.noise_model.realize(self.noise_level.get_power(bandwidth)),
            self.mixer_type,
        )

    @property
    def lo(self) -> RFBlockPort:
        """Local oscillator port."""

        return self.__lo

    @property
    def i(self) -> RFBlockPort:
        """Mixer input port."""

        return self.__i

    @property
    def o(self) -> RFBlockPort:
        """Mixer output port."""

        return self.__o

    @override
    def _propagate(self, realization: MixerRealization, input: RFSignal) -> RFSignal:
        # Compute the resulting frequency of the output signal
        f_lo = input.carrier_frequencies[1]
        f_in = input.carrier_frequencies[0]
        f_out = f_in + f_lo if realization.mixer_type == MixerType.UP else f_in - f_lo

        # Create the output signal
        output: RFSignal = (  # type: ignore
            input[0:1, :] * input[1:2, :]  # type: ignore
            if realization.mixer_type == MixerType.UP
            else input[0, :] * input[1, :].conj()
        )
        output.carrier_frequencies[0] = f_out
        output.carrier_frequency = f_out
        return output

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object(self.__type, "type")
        process.serialize_object(self.noise_model, "noise_model")
        process.serialize_object(self.noise_level, "noise_level")
        if self.seed is not None:
            process.serialize_integer(self.seed, "seed")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> Mixer:
        return Mixer(
            process.deserialize_object("type", MixerType),
            process.deserialize_object("noise_model", NoiseModel),
            process.deserialize_object("noise_level", NoiseLevel),
            process.deserialize_integer("seed", None),
        )


class IdealMixerRealization(MixerRealization):

    __lo_frequency: float

    def __init__(
        self,
        sampling_rate: float,
        oversampling_factor: int,
        noise_realization,
        type: MixerType,
        lo_frequency: float,
    ) -> None:
        """
        Args:
            sampling_rate: Sampling rate of the block in Hz.
            oversampling_factor: Oversampling factor of the modeling.
            noise_realization: Thermal noise realization applied to the mixer.
            type: Type of the mixer, either UP or DOWN.
            lo_frequency: Center frequency of the mixer's oscillator in Hz.
        """

        # Initialize base class
        MixerRealization.__init__(self, sampling_rate, oversampling_factor, noise_realization, type)

        # Store attributes
        self.__lo_frequency = lo_frequency

    @property
    def lo_frequency(self) -> float:
        """Center frequency of the local oscillator in Hz."""

        return self.__lo_frequency


class IdealMixer(RFBlock):
    """An ideal mixer moving the input signal to a different center frequency."""

    __type: MixerType
    __lo_frequency: float
    __i: RFBlockPort[IdealMixer]
    __o: RFBlockPort[IdealMixer]

    def __init__(
        self,
        mixer_type: MixerType,
        lo_frequency: float = 0.0,
        noise_model: NoiseModel | None = None,
        noise_level: NoiseLevel | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            mixer_type: Type of the mixer, either UP or DOWN.
            lo_frequency:
                Center frequency of the mixer's oscillator in Hz.
                If set to zero, the device's configured carrier frequency will be assumed.
            noise_model: Noise model to use for the mixer.
            noise_level: Noise level to use for the mixer.
            seed: Seed with which to initialize the block's random state.
        """

        # Initialize base class
        RFBlock.__init__(self, noise_model, noise_level, seed)

        # Store attributes
        self.mixer_type = mixer_type
        self.lo_frequency = lo_frequency
        self.__i = RFBlockPort(self, 0, RFBlockPortType.IN)
        self.__o = RFBlockPort(self, 0, RFBlockPortType.OUT)

    @property
    @override
    def num_input_ports(self) -> int:
        return 1

    @property
    @override
    def num_output_ports(self) -> int:
        return 1

    @property
    def mixer_type(self) -> MixerType:
        """Type of the mixer."""

        return self.__type

    @mixer_type.setter
    def mixer_type(self, value: MixerType) -> None:
        self.__type = value

    @property
    def lo_frequency(self) -> float:
        """Center frequency of the local oscillator in Hz."""
        return self.__lo_frequency

    @lo_frequency.setter
    def lo_frequency(self, value: float) -> None:
        if value < 0:
            raise ValueError("Local oscillator frequency must be non-negative")
        self.__lo_frequency = value

    @override
    def realize(
        self, bandwidth: float, oversampling_factor: int, carrier_frequency: float
    ) -> IdealMixerRealization:
        return IdealMixerRealization(
            bandwidth,
            oversampling_factor,
            self.noise_model.realize(self.noise_level.get_power(bandwidth)),
            self.mixer_type,
            self.lo_frequency if self.lo_frequency > 0.0 else carrier_frequency,
        )

    @override
    def _propagate(self, realization: IdealMixerRealization, input: RFSignal) -> RFSignal:
        # Compute the resulting frequency of the output signal
        f_in = input.carrier_frequencies[0]
        f_out = (
            abs(f_in + realization.lo_frequency)
            if realization.mixer_type == MixerType.UP
            else abs(f_in - realization.lo_frequency)
        )

        # Create the output signal
        output = input[[0], :].copy()
        output.carrier_frequencies[0] = f_out
        output.carrier_frequency = f_out
        return output

    @property
    def i(self) -> RFBlockPort[IdealMixer]:
        """Input port of the mixer."""

        return self.__i

    @property
    def o(self) -> RFBlockPort[IdealMixer]:
        """Output port of the mixer."""

        return self.__o

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object(self.__type, "type")
        process.serialize_floating(self.__lo_frequency, "lo_frequency")
        process.serialize_object(self.noise_model, "noise_model")
        process.serialize_object(self.noise_level, "noise_level")
        if self.seed is not None:
            process.serialize_integer(self.seed, "seed")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> IdealMixer:
        return IdealMixer(
            process.deserialize_object("type", MixerType),
            process.deserialize_floating("lo_frequency"),
            process.deserialize_object("noise_model", NoiseModel),
            process.deserialize_object("noise_level", NoiseLevel),
            process.deserialize_integer("seed", None),
        )
