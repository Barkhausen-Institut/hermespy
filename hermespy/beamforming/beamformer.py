# -*- coding: utf-8 -*-
"""
==========
Beamformer
==========

Beamforming is split into the prototype classes :class:`.TransmitBeamformer` and :class:`.ReceiveBeamformer`
for beamforming operations during signal transmission and reception, respectively.
They are both derived from the base :class:`BeamformerBase`.
This is due to the fact that some beamforming algorithms may be exclusive to transmission or reception use-cases.
Should a beamformer be applicable during both transmission and reception both prototypes can be inherited.
An example for such an implementation is the :class:`Conventional <.conventional.ConventionalBeamformer>` beamformer.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal, overload, Sequence, TypeVar

import numpy as np

from hermespy.core import (
    AntennaArrayState,
    Direction,
    Operator,
    ReceiveState,
    Serializable,
    Signal,
    State,
    TransmitState,
    TransmitStreamEncoder,
    ReceiveStreamDecoder,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


FT = TypeVar("FT", bound="BeamFocus")
"""Type of beam focus."""


class BeamFocus(ABC, Serializable):
    """Single focus point of a beamformer."""

    property_blacklist = {"beamformer", "spherical_angles"}

    def __init__(self) -> None:
        # Initilize base class
        Serializable.__init__(self)

    @abstractmethod
    def spherical_angles(self, device: State) -> np.ndarray:
        """Compute azimuth and zenith angles in radians, towards a beam is focused.

        Args:

            device (State):
                State of the device the beamformer is operating on.

        Returns:
            Numpy vector with the azimuth and zenith angles in radians from the device's point of view.
        """
        ...  # pragma: no cover

    @abstractmethod
    def copy(self: FT) -> FT:
        """Create a copy of this focus point.

        Returns:

            A copy of this focus point.
        """
        ...  # pragma: no cover


class SphericalFocus(BeamFocus):
    """Focus point in spherical coordinates."""

    yaml_tag = "SphericalFocus"
    __angles: np.ndarray

    @overload
    def __init__(self, angles: np.ndarray) -> None:
        """
        Args:

            angles (numpy.ndarray): Spherical angles in radians.

        Raises:

            ValueError: On invalid argument types.
        """
        ...  # pragma: no cover

    @overload
    def __init__(self, azimuth: float, zenith: float) -> None:
        """
        Args:

            azimuth (float): Azimuth angle in radians.
            zenith (float): Zenith angle in radians.
        """
        ...  # pragma: no cover

    def __init__(  # type: ignore
        self, angles: float | np.ndarray, zenith: float | None = None
    ) -> None:
        if isinstance(angles, (float, int, np.float64, np.int_)):
            self.__angles = np.array([angles, zenith], dtype=np.float64)

        elif isinstance(angles, np.ndarray):
            self.__angles = angles

        else:
            raise ValueError("Invalid argument type")

        # Initialize base class
        BeamFocus.__init__(self)

    @property
    def angles(self) -> np.ndarray:
        """Spherical azimuth and zenith angles in radians."""

        return self.__angles

    def copy(self) -> SphericalFocus:
        return SphericalFocus(self.__angles.copy())

    def spherical_angles(self, device: State) -> np.ndarray:
        return self.__angles


class CoordinateFocus(BeamFocus):
    """Focus the beamformer towards a certain Cartesian coordinate."""

    yaml_tag = "CoordinateFocus"
    __direction: Direction
    __reference: Literal["global", "local"]

    def __init__(
        self, coordinates: np.ndarray | Direction, reference: Literal["global", "local"] = "local"
    ) -> None:
        """
        Args:

            coordinates (numpy.ndarray): Cartesian coordinates in m.
            reference (str, optional): Reference frame of the coordinates.
        """

        # Initialize base class
        BeamFocus.__init__(self)

        # Initialize class members
        self.__direction = (
            coordinates
            if isinstance(coordinates, Direction)
            else Direction.From_Cartesian(coordinates, True)
        )
        self.__reference = reference

    @property
    def coordinates(self) -> np.ndarray:
        """Cartesian coordinates in m."""

        return self.__direction.view(np.ndarray)

    @property
    def reference(self) -> Literal["global", "local"]:
        """Reference frame of the coordinates."""

        return self.__reference

    def copy(self) -> CoordinateFocus:
        return CoordinateFocus(self.__direction.copy(), self.__reference)

    def spherical_angles(self, device: State) -> np.ndarray:
        if self.reference == "local":
            return self.__direction.to_spherical()

        elif self.reference == "global":
            transformation = device.antennas.backwards_transformation
            local_direction = transformation.transform_direction(self.__direction)
            return local_direction.to_spherical()


OT = TypeVar("OT", bound=Operator)
"""Type of operator."""


class TransmitBeamformer(TransmitStreamEncoder):
    """Base class for beam steering precodings during signal transmissions."""

    __focus_points: Sequence[BeamFocus]

    def __init__(self) -> None:
        # Initialize base class
        TransmitStreamEncoder.__init__(self)

        # Initialize attributes
        self.__focus_points = [
            SphericalFocus(0.0, 0.0) for _ in range(self.num_transmit_focus_points)
        ]

    @property
    @abstractmethod
    def num_transmit_focus_points(self) -> int:
        """Number of required transmit focus points.

        If this is :math:`1`,
        the beamformer is considered to be a single focus point beamformer and
        :attr:`.transmit_focus` will return a single focus point.
        Otherwise, the beamformer is considered a multi focus point beamformer and
        :attr:`.transmit_focus` will return a :py:obj:`Sequence` of focus points.
        """
        ...  # pragma: no cover

    @property
    def transmit_focus(self) -> BeamFocus | Sequence[BeamFocus]:
        """Focus points of the beamformer during transmission.

        Depending on :attr:`.num_transmit_focus_points`
        this is either a single focus point or a :py:obj:`Sequence` of points.

        Raises:

            ValueError: If the provided number of focus points does not match the number of required focus points.
        """

        return self.__focus_points

    @transmit_focus.setter
    def transmit_focus(self, value: BeamFocus | Sequence[BeamFocus]) -> None:
        # Force value to be a sequence internally
        _value = [value] if isinstance(value, BeamFocus) else [v for v in value]

        # Make sure the expected number of focus points is provided
        if len(_value) != self.num_transmit_focus_points:
            raise ValueError(
                f"The provided focus contains {len(_value)} points, but the beamformer requires {self.num_transmit_focus_points} focus points"
            )

        # Save results
        self.__focus_points = _value

    def encode_streams(
        self,
        streams: Signal,
        num_output_streams: int,
        device: TransmitState,
        focus: BeamFocus | Sequence[BeamFocus] | None = None,
    ) -> Signal:
        """Encode a MIMO signal for transmit beamforming.

        Wrapper around :meth:`_encode` to encode a signal for transmit beamforming.
        Compliant with the :class:`TransmitStreamEncoder` interface.

        Args:

            streams (Signal):
                The signal to be encoded.

            num_output_streams (int):
                The number of desired output streams.
                Must match :attr:`.num_transmit_output_streams`.

            device (TransmitState):
                State of the device this beamformer is operating on.

            focus (BeamFocus | Sequence[BeamFocus], optional):
                Focus points of the steered signal power.
                Must either be a single focus point or a sequence of focus points,
                depending on :attr:`.num_transmit_focus_points`.
                If not provided, the beamformer's default :attr:`.transmit_focus` is assumed.

        Raises:

            ValueError: If the number of provided signal streams does not match the beamformer requirements.
            ValueError: If the number of provided focus points does not match the beamformer requirements.

        Returns: The encoded signal.
        """

        # Infer the required focus points
        _focus: Sequence[BeamFocus]
        if focus is None:
            _focus = (
                [self.transmit_focus]
                if isinstance(self.transmit_focus, BeamFocus)
                else self.transmit_focus
            )
        elif isinstance(focus, BeamFocus):
            _focus = [focus]
        else:
            _focus = focus

        # Assert the correct number of input signal streams
        num_input_streams = self._num_transmit_input_streams(num_output_streams)
        if streams.num_streams != num_input_streams:
            raise ValueError(
                "Stream encoding configuration invalid, number of provided streams don't match the beamformer requirements"
            )

        # Assert the correct number of focus points
        if len(_focus) != self.num_transmit_focus_points:
            raise ValueError(
                f"The provided focus contains {len(_focus)} points, but the beamformer requires {self.num_transmit_focus_points} focus points"
            )

        # Encode the signal
        steered_samples = self._encode(
            streams.getitem(),
            device.carrier_frequency,
            np.array([focus.spherical_angles(device) for focus in _focus], dtype=np.float64),
            device.antennas,
        )
        return streams.from_ndarray(steered_samples)

    @abstractmethod
    def _encode(
        self,
        samples: np.ndarray,
        carrier_frequency: float,
        focus_angles: np.ndarray,
        array: AntennaArrayState,
    ) -> np.ndarray:
        """Encode signal streams for transmit beamforming.

        Args:

            samples (numpy.ndarray):
                Signal samples, first dimension being the number of transmit antennas, second the number of samples.

            carrier_frequency (float):
                The assumed central carrier frequency of the samples generated RF signal after mixing in Hz.

            focus_angles (numpy.ndarray):
                Focused angles of departure in radians.
                Two-dimensional numpy array with the first dimension representing the number of focus points
                and the second dimension of magnitude two being the azimuth and elevation angles, respectively.

            array (AntennaArrayState):
                The assumed antenna array.

        Returns:
            The encoded signal samples.
            Two-dimensional complex-valued numpy array with the first dimension representing the number of transmit antennas streams
            and the second dimension the number of samples.
        """
        ...  # pragma: no cover


class ReceiveBeamformer(ReceiveStreamDecoder):
    """Base class for beam steering precodings during signal receptions.

    The beamformer is characterised by its required number of input streams :math:`N`,
    the resulting number of output streams :math:`M` and the supported number of focus points :math:`F`.
    Considering a beamformer is provided with a matrix of :math:`T` baseband samples
    :math:`\\mathbf{X} \\in \\mathbb{C}^{N \\times T}`, the beamforming process

    .. math::

       \\mathbf{Y} = \\mathcal{B}\\lbrace \\mathbf{X} \\rbrace \\quad \\text{with} \\quad \\mathbf{Y} \\in \\mathbb{C}^{M \\times T}

    can generally be described as a function compressing the number of streams while focusing the power towards
    the angles of interest :math:`F`.
    """

    __focus_points: Sequence[BeamFocus]

    def __init__(self) -> None:
        # Initialize base class
        ReceiveStreamDecoder.__init__(self)

        # Initialize attributes
        self.receive_focus = [
            SphericalFocus(0.0, 0.0) for _ in range(self.num_receive_focus_points)
        ]
        self.probe_focus_points = np.zeros((1, self.num_receive_focus_points, 2), dtype=float)

    @property
    @abstractmethod
    def num_receive_focus_points(self) -> int:
        """Number of required receive focus points.

        If this is :math:`1`,
        the beamformer is considered to be a single focus point beamformer and
        :attr:`.receive_focus` will return a single focus point.
        Otherwise, the beamformer is considered a multi focus point beamformer and
        :attr:`.receive_focus` will return a :py:obj:`Sequence` of focus points.

        Returns: Number of focus points.
        """
        ...  # pragma: no cover

    @property
    def receive_focus(self) -> BeamFocus | Sequence[BeamFocus]:
        """Focus points of the beamformer during reception.

        Depending on :attr:`.num_receive_focus_points`
        this is either a single focus point or a :py:obj:`Sequence` of points.

        Raises:

            ValueError: If the provided number of focus points does not match the number of required focus points.
        """

        return self.__focus_points

    @receive_focus.setter
    def receive_focus(self, value: BeamFocus | Sequence[BeamFocus]) -> None:
        # Force value to be a sequence internally
        _value = [value] if isinstance(value, BeamFocus) else [v for v in value]

        # Make sure the expected number of focus points is provided
        if len(_value) != self.num_receive_focus_points:
            raise ValueError(
                f"The provided focus contains {len(_value)} points, but the beamformer requires {self.num_receive_focus_points} focus points"
            )

        # Save results
        self.__focus_points = _value

    @property
    def probe_focus_points(self) -> np.ndarray:
        """Default beamformer focus points during probing.

        Returns:

            The focus points as a three-dimensional numpy array, with the first dimension
            representing the probe index, the second dimension the point and the third dimension of magnitude
            two the point azimuth and zenith, respectively.

        Raises:

            ValueError: On invalid arguments.
        """

        return self.__probe_focus_points

    @probe_focus_points.setter
    def probe_focus_points(self, value: np.ndarray) -> None:
        # Expand points by new dimension if only a single focus tuple was requested
        if value.ndim == 2:
            value = value[np.newaxis, ::]

        if value.ndim != 3:
            raise ValueError("Probing focus points must be a three-dimensional array")

        if value.shape[1] != self.num_receive_focus_points:
            raise ValueError(
                f"Focus requires {self.num_receive_focus_points} points, but {value.shape[1]} were provided"
            )

        self.__probe_focus_points = value

    @abstractmethod
    def _decode(
        self,
        samples: np.ndarray,
        carrier_frequency: float,
        angles: np.ndarray,
        array: AntennaArrayState,
    ) -> np.ndarray:
        """Decode signal streams for receive beamforming.

        This method is called as a subroutine during :meth:`decode_streams` and :meth:`probe`.

        Args:

            samples (numpy.ndarray):
                Signal samples, first dimension being the number of signal streams :math:`N`, second the number of samples :math:`T`.

            carrier_frequency (float):
                The assumed carrier central frequency of the samples :math:`f_\\mathrm{c}`.

            angles (numpy.ndarray):
                Spherical coordinate system angles of arrival in radians.
                A three-dimensional numpy array with the first dimension representing the number of angles,
                the second dimension of magnitude number of focus points :math:`F`,
                and the third dimension containing the azimuth and zenith angle in radians, respectively.

            array (AntennaArrayState):
                The assumed antenna array.

        Returns:

            Stream samples of the focused signal towards all focus points.
            A three-dimensional numpy array with the first dimension representing the number of focus points,
            the second dimension the number of returned streams and the third dimension the amount of samples.
        """
        ...  # pragma: no cover

    def decode_streams(
        self,
        streams: Signal,
        num_output_streams: int,
        device: ReceiveState,
        focus: BeamFocus | Sequence[BeamFocus] | None = None,
    ) -> Signal:
        """Decode a MIMO signal for receive beamforming.

        Wrapper around :meth:`_decode` to decode a signal for receive beamforming.
        Compliant with the :class:`ReceiveStreamDecoder` interface.

        Args:

            streams (Signal):
                The signal to be decoded.

            num_output_streams (int):
                The number of desired output streams.
                Must match :attr:`.num_receive_output_streams`.

            device (ReceiveState):
                State of the device this beamformer is operating on.

            focus (BeamFocus | Sequence[BeamFocus], optional):
                Focus points of the steered signal power.
                Must either be a single focus point or a sequence of focus points,
                depending on :attr:`.num_receive_focus_points`.
                If not provided, the beamformer's default :attr:`.receive_focus` is assumed.

        Raises:

            ValueError:
                - If the number of provided focus points does not match the beamformer requirements.

        Returns: The decoded signal.
        """

        # Infer the required focus points
        _focus: Sequence[BeamFocus]
        if focus is None:
            _focus = (
                [self.receive_focus]
                if isinstance(self.receive_focus, BeamFocus)
                else self.receive_focus
            )
        elif isinstance(focus, BeamFocus):
            _focus = [focus]
        else:
            _focus = focus

        # Assert the correct number of focus points
        if len(_focus) != self.num_receive_focus_points:
            raise ValueError(
                f"The provided focus contains {len(_focus)} points, but the beamformer requires {self.num_receive_focus_points} focus points"
            )

        # Decode the signal
        beamformed_samples = self._decode(
            streams.getitem(),
            device.carrier_frequency,
            np.array([[focus.spherical_angles(device) for focus in _focus]], dtype=np.float64),
            device.antennas,
        )
        return streams.from_ndarray(beamformed_samples[0, ::])

    def probe(
        self, signal: Signal, device: ReceiveState, focus_points: np.ndarray | None = None
    ) -> np.ndarray:
        """Focus a signal model towards certain directions of interest.

        Args:

            signal (Signal):
                The signal to be steered.

            device (ReceiveState):
                State of the device this beamformer is operating on.

            focus_points (numpy.ndarray, optional):
                Focus point of the steered signal power.
                Two-dimensional numpy array with the first dimension representing the number of points
                and the second dimension representing the point values.

        Returns:

            Stream samples of the focused signal towards all focus points.
            A three-dimensional numpy array with the first dimension representing the number of focus points,
            the second dimension the number of returned streams and the third dimension the amount of samples.
        """

        # Infer the required focus points
        _focus_points = self.probe_focus_points if focus_points is None else focus_points

        # Decode the signal focusing towards the provided points
        return self._decode(
            signal.getitem(), device.carrier_frequency, _focus_points, device.antennas
        )
