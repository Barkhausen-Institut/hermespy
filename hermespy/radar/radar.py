# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod
from typing import Generic, Tuple, Type, TypeVar

import numpy as np
from h5py import Group
from scipy.constants import speed_of_light

from hermespy.beamforming import ReceiveBeamformer, TransmitBeamformer
from hermespy.core import (
    Device,
    DuplexOperator,
    FloatingError,
    Signal,
    Serializable,
    Transmission,
    Reception,
)
from .cube import RadarCube
from .detection import RadarDetector, RadarPointCloud

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarWaveform(object):
    """Base class for :class:`.Radar` waveform descriptions.

    When assigned to a :class:`.Radar`'s :meth:`waveform<Radar.waveform>` property,
    the waveform's :meth:`.ping` and :meth:`.estimate` methods are called as subroutines
    of the :class:`.Radar`'s :meth:`Radar.transmit` and :meth:`Radar.receive` routines, respectively.


    .. mermaid::

        classDiagram

            class Radar {
                +waveform : RadarWaveform
                +transmit() : RadarTransmission
                +receive() : RadarReception
            }

            class RadarWaveform {
                <<Abstract>>
                +sampling_rate: float*
                +frame_duration: float*
                +max_range: float*
                +range_resolution: float*
                +range_bins: ndarray
                +max_relative_doppler: float*
                +relative_doppler_resolution: float*
                +energy: float*
                +power: float*
                +ping() : Signal*
                +estimate(Signal) : ndarray*

            }

            class FMCW {
                +sampling_rate: float
                +frame_duration: float
                +max_range: float
                +range_resolution: float
                +range_bins: ndarray
                +max_relative_doppler: float
                +relative_doppler_resolution: float
                +energy: float
                +power: float
                +ping() : Signal
                +estimate(Signal) : ndarray

            }

            Radar *-- RadarWaveform
            FMCW ..|> RadarWaveform
            Radar --> RadarWaveform: ping()
            Radar --> RadarWaveform: estimate()

            link RadarWaveform "#hermespy.radar.radar.RadarWaveform"
            link Radar "radar.radar.Radar.html"
            link FMCW "radar.fmcw.html"

    The currently available radar waveforms are:

    .. toctree::

        radar.fmcw


    """

    @abstractmethod
    def ping(self) -> Signal:
        """Generate a single radar frame.

        Returns:
            Single-stream signal model of a single radar frame.
        """
        ...  # pragma: no cover

    @abstractmethod
    def estimate(self, signal: Signal) -> np.ndarray:
        """Generate a range-doppler map from a single-stream radar frame.

        Args:

            signal (Signal): Single-stream signal model of a single propagated radar frame.

        Returns:
            Numpy matrix (2D array) of the range-doppler map, where the first dimension indicates
            discrete doppler frequency bins and the second dimension indicates discrete range bins.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def sampling_rate(self) -> float:
        """The optional sampling rate required to process this waveform.

        Denoted by :math:`f_\\mathrm{s}` of unit :math:`\\left[ f_\\mathrm{s} \\right] = \\mathrm{Hz} = \\tfrac{1}{\\mathrm{s}}` in literature.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def frame_duration(self) -> float:
        """Duration of a single radar frame in seconds.

        Denoted by :math:`T_{\\mathrm{F}}` of unit :math:`\\left[ T_{\\mathrm{F}} \\right] = \\mathrm{s}` in literature.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def max_range(self) -> float:
        """The waveform's maximum detectable range in meters.

        Denoted by :math:`R_{\\mathrm{Max}}` of unit :math:`\\left[ R_{\\mathrm{Max}} \\right] = \\mathrm{m}` in literature.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def range_resolution(self) -> float:
        """Resolution of the radial range sensing in meters.

        Denoted by :math:`\Delta R` of unit :math:`\\left[ \Delta R \\right] = \\mathrm{m}` in literature.
        """
        ...  # pragma: no cover

    @property
    def range_bins(self) -> np.ndarray:
        """Discrete sample bins of the radial range sensing.

        Returns:
            A numpy vector (1D array) of discrete range bins in meters.
        """

        return np.arange(int(self.max_range / self.range_resolution)) * self.range_resolution

    @property
    @abstractmethod
    def max_relative_doppler(self) -> float:
        """Maximum relative detectable radial doppler frequency shift in Hz.

        .. math::

           \Delta f_\\mathrm{Max} = \\frac{v_\\mathrm{Max}}{\\lambda}

        Returns: Shift frequency delta in Hz.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def relative_doppler_resolution(self) -> float:
        """Relative resolution of the radial doppler frequency shift sensing in Hz.

        .. math::

           \Delta f_\\mathrm{Res} = \\frac{v_\\mathrm{Res}}{\\lambda}

        Returns: Doppler resolution in Hz.
        """
        ...  # pragma: no cover

    @property
    def relative_doppler_bins(self) -> np.ndarray:
        """Realtive discrete sample bins of the radial doppler frequency shift sensing.

        Returns:
            A numpy vector (1D array) of discrete doppler frequency bins in Hz.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def energy(self) -> float:
        """Energy of the radar waveform.

        Returns: Radar energy in :math:`\\mathrm{Wh}`.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def power(self) -> float:
        """Power of the radar waveform.

        Returns: Radar power in :math:`\\mathrm{W}`.
        """
        ...  # pragma: no cover


class RadarTransmission(Transmission):
    """Information generated by transmitting over a radar.
    Generated by calling a :class:`.Radar`'s :meth:`transmit()<.Radar.transmit>` method.
    """

    def __init__(self, signal: Signal) -> None:
        """
        Args:

            signal (Signal): Transmitted radar waveform.
        """

        Transmission.__init__(self, signal)


class RadarReception(Reception):
    """Information generated by receiving over a radar.

    Generated by calling a :class:`.Radar`'s :meth:`receive()<.Radar.receive>` method.
    """

    __cube: RadarCube  # Processed raw radar data
    __cloud: RadarPointCloud | None  # Detected radar point cloud

    def __init__(
        self, signal: Signal, cube: RadarCube, cloud: RadarPointCloud | None = None
    ) -> None:
        """
        Args:

            signal (Signal): Received radar waveform.
            cube (RadarCube): Processed raw radar data.
            cloud (RadarPointCloud, optional): Radar point cloud. :obj:`None` if a point cloud was not generated.
        """

        # Initialize base class
        Reception.__init__(self, signal)

        # Initialize class attributes
        self.__cube = cube
        self.__cloud = cloud

    @property
    def cube(self) -> RadarCube:
        """Cube of processed raw radar data."""

        return self.__cube

    @property
    def cloud(self) -> RadarPointCloud | None:
        """Detected radar point cloud.

        :obj:`None` if a point cloud was not generated."""

        return self.__cloud

    def to_HDF(self, group: Group) -> None:
        # Serialize base class
        Reception.to_HDF(self, group)

        # Serialize class attributes
        self.cube.to_HDF(self._create_group(group, "cube"))
        return

    @classmethod
    def from_HDF(cls: Type[RadarReception], group: Group) -> RadarReception:
        signal = Signal.from_HDF(group["signal"])
        cube = RadarCube.from_HDF(group["cube"])

        return RadarReception(signal, cube)


RTT = TypeVar("RTT", bound=RadarTransmission)
"""Type of radar transmission."""

RRT = TypeVar("RRT", bound=RadarReception)
"""Type of radar reception."""


class RadarBase(Generic[RTT, RRT], DuplexOperator[RTT, RRT]):
    """Base class class for radar sensing signal processing pipelines."""

    __transmit_beamformer: TransmitBeamformer | None
    __receive_beamformer: ReceiveBeamformer | None
    __detector: RadarDetector | None

    def __init__(self, *args, **kwargs) -> None:
        # Initialize base class
        DuplexOperator.__init__(self, *args, **kwargs)

        # Initialize class attributes
        self.receive_beamformer = None
        self.transmit_beamformer = None
        self.detector = None

    @property
    def transmit_beamformer(self) -> TransmitBeamformer | None:
        """Beamforming applied during signal transmission.

        The :class:`TransmitBeamformer<hermespy.beamforming.beamformer.TransmitBeamformer>`'s :meth:`transmit<hermespy.beamforming.beamformer.TransmitBeamformer.transmit>`
        method is called as a subroutine of :meth:`Transmitter.transmit()<hermespy.core.device.Transmitter.transmit>`.
        Configuration is required for if the assigned :class:`Device<hermespy.core.device.Device>` features multiple :meth:`antennas<hermespy.core.device.Device.antennas>`.
        """

        return self.__transmit_beamformer

    @transmit_beamformer.setter
    def transmit_beamformer(self, value: TransmitBeamformer | None) -> None:
        if value is None:
            self.__transmit_beamformer = None

        else:
            value.operator = self
            self.__transmit_beamformer = value

    @property
    def receive_beamformer(self) -> ReceiveBeamformer | None:
        """Beamforming applied during signal reception.

        The :class:`TransmitBeamformer<hermespy.beamforming.beamformer.ReceiveBeamformer>`'s :meth:`receive<hermespy.beamforming.beamformer.ReceiveBeamformer.receive>`
        method is called as a subroutine of :meth:`Receiver.receive()<hermespy.core.device.Receiver.receive>`.
        Configuration is required for if the assigned :class:`Device<hermespy.core.device.Device>` features multiple :meth:`antennas<hermespy.core.device.Device.antennas>`.
        """

        return self.__receive_beamformer

    @receive_beamformer.setter
    def receive_beamformer(self, value: ReceiveBeamformer | None) -> None:
        if value is None:
            self.__receive_beamformer = None

        else:
            value.operator = self
            self.__receive_beamformer = value

    @property
    def detector(self) -> RadarDetector | None:
        """Detector routine configured to generate point clouds from radar cubes.

        If configured, during :meth:`_receive<Radar._receive>` / :meth:`receive<Receiver.receive>`,
        the detector's :meth:`detect<RadarDetector.detect>` method is called to generate a :class:`RadarPointCloud<hermespy.radar.detection.RadarPointCloud>`.
        If not configured, i.e. :obj:`None`, the generated :class:`.RadarReception`'s :attr:`cloud<.RadarReception.cloud>` property will be :obj:`None`.
        """

        return self.__detector

    @detector.setter
    def detector(self, value: RadarDetector | None) -> None:
        self.__detector = value

    def _receive_beamform(self, signal: Signal) -> Tuple[np.ndarray, np.ndarray]:
        """Apply digital beamforming to the received signal.

        Subroutine of :meth:`_receive<Radar._receive>`.

        Args:

            signal (Signal): Received radar waveform.

        Returns:
            Tuple of the angles of interest and the beamformed samples.

        Raises:

            RuntimeError: If the device has more than one antenna, but no beamforming strategy is configured.
            RuntimeError: If the beamforming configuration does not result in a single output stream.
        """

        if self.device.antennas.num_antennas > 1:
            if self.receive_beamformer is None:
                raise RuntimeError(
                    "Receiving over a device with more than one antenna requires a beamforming configuration"
                )

            if self.receive_beamformer.num_receive_output_streams != 1:
                raise RuntimeError(
                    "Only receive beamformers generating a single output stream are supported by radar operators"
                )

            if (
                self.receive_beamformer.num_receive_input_streams
                != self.device.antennas.num_antennas
            ):
                raise RuntimeError(
                    "Radar operator receive beamformers are required to consider the full number of antenna streams"
                )

            beamformed_samples = self.receive_beamformer.probe(signal)[:, 0, :]

        else:
            beamformed_samples = signal[:, :]

        # Build the radar cube by generating a beam-forming line over all angles of interest
        angles_of_interest = (
            np.array([[0.0, 0.0]], dtype=float)
            if self.receive_beamformer is None
            else self.receive_beamformer.probe_focus_points[:, 0, :]
        )

        return angles_of_interest, beamformed_samples


class Radar(RadarBase[RadarTransmission, RadarReception], Serializable):
    """Signal processing pipeline of a monostatic radar sensing its environment.

    The radar can be configured by assigning four composite objects to respective property attributes:

    .. list-table::
       :header-rows: 1

       * - Property
         - Type

       * - :meth:`waveform<.waveform>`
         - :class:`RadarWaveform`

       * - :meth:`transmit_beamformer<.transmit_beamformer>`
         - :class:`TransmitBeamformer<hermespy.beamforming.beamformer.TransmitBeamformer>`

       * - :meth:`receive_beamformer<.receive_beamformer>`
         - :class:`ReceiveBeamformer<hermespy.beamforming.beamformer.ReceiveBeamformer>`

       * - :meth:`detector<.detector>`
         - :class:`RadarDetector<hermespy.radar.detection.RadarDetector>`

    Of those, only a :class:`RadarWaveform` is mandatory.
    Beamformers, i.e. a :class:`TransmitBeamformer<hermespy.beamforming.beamformer.TransmitBeamformer>` and a :class:`ReceiveBeamformer<hermespy.beamforming.beamformer.ReceiveBeamformer>`,
    are only required when the radar is assigned to a :class:`Device<hermespy.core.device.Device>` configured to multiple :meth:`antennas<hermespy.core.device.Device.antennas>`.
    A :class:`RadarDetector<hermespy.radar.detection.RadarDetector>` is optional,
    if not configured the radar's generated :class:`RadarReception` will not contain a :class:`RadarPointCloud<hermespy.radar.detection.RadarPointCloud>`.

    When assigned to a :class:`Device<hermespy.core.device.Device>`,
    device transmission will trigger the radar to generate a :class:`RadarTransmission` by executing the following sequence of calls:

    .. mermaid::

        sequenceDiagram

        participant Device
        participant Radar
        participant RadarWaveform
        participant TransmitBeamformer

        Device ->> Radar: _transmit()
        Radar ->> RadarWaveform: ping()
        RadarWaveform -->> Radar: Signal
        Radar ->>  TransmitBeamformer: transmit(Signal)
        TransmitBeamformer -->> Radar: Signal
        Radar -->> Device: RadarTransmission

    Initially, the :meth:`ping<RadarWaveform.ping>` method of the :class:`RadarWaveform` is called to generate the model
    of a single-antenna radar frame.
    For :class:`Devices<hermespy.core.device.Device>` configured to multiple :meth:`antennas<hermespy.core.device.Device.antennas>`,
    the configured :class:`TransmitBeamformer<hermespy.beamforming.beamformer.TransmitBeamformer>` is called to encode the signal for each antenna.
    The resulting multi-antenna frame, contained within the return :class:`RadarTransmission`, is cached at the assigned :class:`Device<hermespy.core.device.Device>`.

    When assigned to a :class:`Device<hermespy.core.device.Device>`,
    device reception will trigger the radar to generate a :class:`RadarReception` by executing the following sequence of calls:

    .. mermaid::

        sequenceDiagram

            participant Device
            participant Radar
            participant ReceiveBeamformer
            participant RadarWaveform
            participant RadarDetector

            Device ->> Radar: _receive(Signal)
            Radar ->> ReceiveBeamformer: probe(Signal)
            ReceiveBeamformer -->> Radar: line_signals
            loop
                Radar ->> RadarWaveform: estimate(line_signal)
                RadarWaveform -->> Radar: line_estimate
            end
            Radar ->> RadarDetector: detect(line_estimates)
            RadarDetector -->> Radar: RadarPointCloud
            Radar -->> Device: RadarReception

    Initially, the :meth:`probe<hermespy.beamforming.beamformer.ReceiveBeamformer.probe>` method of the :class:`ReceiveBeamformer<hermespy.beamforming.beamformer.ReceiveBeamformer>` is called to generate a sequence of
    line signals from each direction of interest.
    We refer to them as *line signals* as they are the result of an antenna arrays beamforing towards a single direction of interest,
    so the signal can be though of as having propagated along a single line pointing towards the direction of interest.
    This step is only executed for :class:`Devices<hermespy.core.device.Device>` configured to multiple :meth:`antennas<hermespy.core.device.Device.antennas>`.
    The sequence of line signals are then indiviually processed by the :meth:`estimate<RadarWaveform.estimate>` method of the :class:`RadarWaveform`,
    resulting in a line estimate representing a range-doppler map for each direction of interest.
    This sequence line estimates is combined to a single :class:`RadarCube<hermespy.radar.cube.RadarCube>`.
    If a :class:`RadarDetector<hermespy.radar.detection.RadarDetector>` is configured, the :meth:`detect<hermespy.radar.detection.RadarDetector.detect>` method is called to generate a :class:`RadarPointCloud<hermespy.radar.detection.RadarPointCloud>`
    from the :class:`RadarCube<hermespy.radar.cube.RadarCube>`.
    The resulting information is cached as a :class:`RadarReception` at the assigned :class:`Device<hermespy.core.device.Device>`.
    """

    yaml_tag = "Radar"
    property_blacklist = {"slot"}

    __waveform: RadarWaveform | None

    def __init__(self, device: Device | None = None, seed: int | None = None) -> None:
        """
        Args:

            device (Device, optional):
                The device the radar is assigned to.

            seed (int, optional):
                Random seed used to generate the radar's internal state.
        """

        self.waveform = None
        self.__waveform = None

        RadarBase.__init__(self, device, device, seed)

    @property
    def sampling_rate(self) -> float:
        return self.waveform.sampling_rate

    @property
    def frame_duration(self) -> float:
        if self.waveform is None:
            return 0.0

        return self.waveform.frame_duration

    @property
    def power(self) -> float:
        if self.waveform is None:
            return 0.0

        return self.waveform.power

    @property
    def waveform(self) -> RadarWaveform | None:
        """Description of the waveform to be transmitted and received by this radar.

        :obj:`None` if no waveform is configured.

        During :meth:`transmit<Radar.transmit>` / :meth:`_transmit<Radar._transmit>`,
        the :class:`.RadarWaveform`'s :meth:`ping()<.RadarWaveform.ping>` method is called
        to generate a signal to be transmitted by the radar.
        During :meth:`receive<Radar.receive>` / :meth:`_receive<Radar._receive>`, the :class:`.RadarWaveform`'s :meth:`estimate()<.RadarWaveform.estimate>` method is called
        multiple times to generate range-doppler line estimates for each direction of interest.
        """

        return self.__waveform

    @waveform.setter
    def waveform(self, value: RadarWaveform | None) -> None:
        self.__waveform = value

    @property
    def max_range(self) -> float:
        """The radar's maximum detectable range in meters.

        Denoted by :math:`R_{\\mathrm{Max}}` of unit :math:`\\left[ R_{\\mathrm{Max}} \\right] = \\mathrm{m}` in literature.
        Convenience property that resolves to the configured :class:`.RadarWaveform`'s :meth:`max_range<.RadarWaveform.max_range>` property.
        Returns :math:`R_{\\mathrm{Max}} = 0` if no waveform is configured.
        """

        return 0.0 if self.waveform is None else self.waveform.max_range

    @property
    def velocity_resolution(self) -> float:
        """The radar's velocity resolution in meters per second.

        Denoted by :math:`\\Delta v` of unit :math:`\\left[ \\Delta v \\right] = \\frac{\\mathrm{m}}{\\mathrm{s}}` in literature.
        Computed as

        .. math::

            \\Delta v = \\frac{c_0}{f_{\\mathrm{c}}} \\Delta f_{\\mathrm{Res}}

        querying the configured :class:`.RadarWaveform`'s :meth:`relative_doppler_resolution<.RadarWaveform.relative_doppler_resolution>` property :math:`\\Delta f_{\\mathrm{Res}}`.
        """

        if self.waveform is None:
            raise FloatingError("Cannot compute velocity resolution without a waveform")

        if self.carrier_frequency == 0.0:
            raise RuntimeError("Cannot compute velocity resolution in base-band carrier frequency")

        return (
            0.5
            * self.waveform.relative_doppler_resolution
            * speed_of_light
            / self.carrier_frequency
        )

    def _transmit(self, duration: float = 0.0) -> RadarTransmission:
        if not self.__waveform:
            raise RuntimeError("Radar waveform not specified")

        if not self.device:
            raise RuntimeError("Error attempting to transmit over a floating radar operator")

        # Generate the radar waveform
        signal = self.waveform.ping()

        # If the device has more than one antenna, a beamforming strategy is required
        if self.device.antennas.num_antennas > 1:
            # If no beamformer is configured, only the first antenna will transmit the ping
            if self.transmit_beamformer is None:
                additional_streams = signal.from_ndarray(
                    np.zeros(
                        (
                            self.device.antennas.num_antennas - signal.num_streams,
                            signal.num_samples,
                        ),
                        dtype=complex,
                    )
                )
                signal.append_streams(additional_streams)

            elif self.transmit_beamformer.num_transmit_input_streams != 1:
                raise RuntimeError(
                    "Only transmit beamformers requiring a single input stream are supported by radar operators"
                )

            elif (
                self.transmit_beamformer.num_transmit_output_streams
                != self.device.antennas.num_antennas
            ):
                raise RuntimeError(
                    "Radar operator transmit beamformers are required to consider the full number of antennas"
                )

            else:
                signal = self.transmit_beamformer.transmit(signal)

        # Prepare transmission
        signal.carrier_frequency = self.carrier_frequency
        transmission = RadarTransmission(signal)

        return transmission

    def _receive(self, signal: Signal) -> RadarReception:
        if not self.waveform:
            raise RuntimeError("Radar waveform not specified")

        if not self.device:
            raise RuntimeError("Error attempting to receive over a floating radar operator")

        # Resample signal properly
        signal = signal.resample(self.__waveform.sampling_rate)

        # If the device has more than one antenna, a beamforming strategy is required
        angles_of_interest, beamformed_samples = self._receive_beamform(signal)
        range_bins = self.waveform.range_bins
        doppler_bins = self.waveform.relative_doppler_bins

        cube_data = np.empty(
            (len(angles_of_interest), len(doppler_bins), len(range_bins)), dtype=float
        )

        for angle_idx, line in enumerate(beamformed_samples):
            # Process the single angular line by the waveform generator
            line_signal = signal.from_ndarray(line)
            line_estimate = self.waveform.estimate(line_signal)

            cube_data[angle_idx, ::] = line_estimate

        # Create radar cube object
        cube = RadarCube(
            cube_data, angles_of_interest, doppler_bins, range_bins, self.carrier_frequency
        )

        # Infer the point cloud, if a detector has been configured
        cloud = None if self.detector is None else self.detector.detect(cube)

        reception = RadarReception(signal, cube, cloud)
        return reception

    def _recall_transmission(self, group: Group) -> RadarTransmission:
        return RadarTransmission.from_HDF(group)

    def _recall_reception(self, group: Group) -> RadarReception:
        return RadarReception.from_HDF(group)
