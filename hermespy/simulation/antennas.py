# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, List, overload, Sequence, Type, Literal

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
from scipy.constants import pi

from hermespy.beamforming import TransmitBeamformer, ReceiveBeamformer
from hermespy.core import (
    Antenna,
    AntennaArray,
    AntennaArrayState,
    AntennaMode,
    AntennaPort,
    CustomAntennaArray,
    Dipole,
    Executable,
    IdealAntenna,
    LinearAntenna,
    PatchAntenna,
    ReceiveState,
    Signal,
    Transformation,
    TransmitState,
    UniformArray,
)
from .coupling import Coupling
from .isolation import Isolation
from .rf_chain import RfChain

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SimulatedAntennaPort(AntennaPort["SimulatedAntenna", "SimulatedAntennaArray"]):
    """Port within a simulated antenna array."""

    __rf_chain: RfChain | None  # radio frequency chain connected to this antenna port

    def __init__(
        self,
        antennas: Sequence[SimulatedAntenna] = None,
        pose: Transformation | None = None,
        rf_chain: RfChain | None = None,
    ) -> None:
        """
        Args:

            antennas:
                Sequence of antennas connected to this antenna port.
                If not specified, an empty sequence is assumed.

            pose:
                The antenna's position and orientation with respect to its array.

            rf_chain:
                The antenna's RF chain.
                If not specified, the connected device's default RF chain is assumed.
        """

        # Initialize base class
        AntennaPort.__init__(self, antennas, pose)

        # Initialize attributes
        self.__rf_chain = None
        self.rf_chain = rf_chain

    @property
    def rf_chain(self) -> RfChain | None:
        """The antenna's RF chain."""

        return self.__rf_chain

    @rf_chain.setter
    def rf_chain(self, value: RfChain | None) -> None:
        # Abort if the RF chain configuration didn't change
        if value == self.__rf_chain:
            return

        # Update the RF chain configuration
        self.__rf_chain = value

        # Notify the antenna array about the change in its RF chain configuration
        if self.array is not None:
            self.array.rf_chain_modified()


class SimulatedAntenna(Antenna[SimulatedAntennaPort]):
    """Model of single antenna within an antenna array."""

    __weight: complex  # Phase and amplitude shift of signals transmitted / received by this antenna

    def __init__(
        self,
        mode: AntennaMode = AntennaMode.DUPLEX,
        pose: Transformation | None = None,
        weight: complex = 1.0 + 0.0j,
    ) -> None:
        """
        Args:

            mode:
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            pose:
                The antenna's position and orientation with respect to its array.

            weight:
                Phase and amplitude shift of signals transmitted and received by this antenna.
                By default, no phase and amplitude shift is applied.
        """

        # Initialize base class
        Antenna.__init__(self, mode, pose)

        # Initialize attributes
        self.weight = weight

    @property
    def weight(self) -> complex:
        """Phase and amplitude shift of signals transmitted and received by this antenna."""

        return self.__weight

    @weight.setter
    def weight(self, value: complex) -> None:
        self.__weight = value

    def transmit(self, signal: Signal) -> Signal:
        """Transmit a signal over this antenna.

        The transmission may be distorted by the antennas impulse response / frequency characteristics.

        Args:

            signal:
                The signal model to be transmitted.

        Returns: The actually transmitted (distorted) signal model.

        Raises:

            ValueError: If the signal has more than one stream.
        """

        if signal.num_streams != 1:
            raise ValueError("Only single-streamed signal can be transmitted over a single antenna")

        if self.weight != 1.0:
            signal = signal.copy()
            for block in signal:
                block *= self.weight

        return signal

    def receive(self, signal: Signal) -> Signal:
        """Receive a signal over this antenna.

        The reception may be distorted by the antennas impulse response / frequency characteristics.

        Args:

            signal:
                The signal model to be received.

        Returns:

            Signal:
                The actually received (distorted) signal model.
        """

        if signal.num_streams != 1:
            raise ValueError("Only single-streamed signal can be received over a single antenna")

        if self.weight != 1.0:
            signal = signal.copy()
            for block in signal:
                block *= self.weight

        return signal


class SimulatedDipole(SimulatedAntenna, Dipole[SimulatedAntennaPort]):
    """Model of single dipole antenna within an antenna array."""

    def __init__(
        self,
        mode: AntennaMode = AntennaMode.DUPLEX,
        pose: Transformation | None = None,
        weight: complex = 1.0 + 0j,
    ) -> None:
        """
        Args:

            mode:
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            pose:
                The antenna's position and orientation with respect to its array.

            weight:
                Phase and amplitude shift of signals transmitted and received by this antenna.
                By default, no phase and amplitude shift is applied.
        """

        # Initialize base classes
        Dipole.__init__(self, mode, pose)
        SimulatedAntenna.__init__(self, mode, pose, weight)


class SimulatedIdealAntenna(SimulatedAntenna, IdealAntenna[SimulatedAntennaPort]):
    """Model of single ideal antenna within an antenna array."""

    def __init__(
        self,
        mode: AntennaMode = AntennaMode.DUPLEX,
        pose: Transformation | None = None,
        weight: complex = 1.0 + 0j,
    ) -> None:
        """
        Args:

            mode:
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            pose:
                The antenna's position and orientation with respect to its array.

            weight:
                Phase and amplitude shift of signals transmitted and received by this antenna.
                By default, no phase and amplitude shift is applied.
        """

        # Initialize base classes
        IdealAntenna.__init__(self, mode, pose)
        SimulatedAntenna.__init__(self, mode, pose, weight)


class SimulatedLinearAntenna(SimulatedAntenna, LinearAntenna[SimulatedAntennaPort]):
    """Model of single linear antenna within an antenna array."""

    def __init__(
        self,
        mode: AntennaMode = AntennaMode.DUPLEX,
        slant: float = 0.0,
        pose: Transformation | None = None,
        weight: complex = 1.0 + 0j,
    ) -> None:
        """
        Args:

            mode:
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            slant:
                The antenna's slant angle in radians.

            pose:
                The antenna's position and orientation with respect to its array.

            weight:
                Phase and amplitude shift of signals transmitted and received by this antenna.
                By default, no phase and amplitude shift is applied.
        """

        # Initialize base classes
        LinearAntenna.__init__(self, mode, slant, pose)
        SimulatedAntenna.__init__(self, mode, pose, weight)


class SimulatedPatchAntenna(SimulatedAntenna, PatchAntenna[SimulatedAntennaPort]):
    """Model of single patch antenna within an antenna array."""

    def __init__(
        self,
        mode: AntennaMode = AntennaMode.DUPLEX,
        pose: Transformation | None = None,
        weight: complex = 1.0 + 0j,
    ) -> None:
        """
        Args:

            mode:
                Antenna's mode of operation.
                By default, a full duplex antenna is assumed.

            pose:
                The antenna's position and orientation with respect to its array.

            weight:
                Phase and amplitude shift of signals transmitted and received by this antenna.
                By default, no phase and amplitude shift is applied.
        """

        # Initialize base classes
        SimulatedAntenna.__init__(self, mode, pose, weight)
        PatchAntenna.__init__(self, mode, pose)


class SimulatedAntennaArray(AntennaArray[SimulatedAntennaPort, SimulatedAntenna]):
    """Array of simulated antennas."""

    __cached_default_rf_chain: RfChain | None
    __cached_rf_transmit_map: Dict[RfChain | None, List[int]] | None
    __cached_rf_receive_map: Dict[RfChain | None, List[int]] | None

    def __init__(self, pose: Transformation | None = None) -> None:
        """
        Args:

            pose:
                The antenna array's position and orientation with respect to its device.
                If not specified, the same orientation and position as the device is assumed.
        """

        # Initialize base class
        AntennaArray.__init__(self, pose)

        # Initialize class attributes
        self.__cached_default_rf_chain = None
        self.__cached_rf_transmit_map = None
        self.__cached_rf_receive_map = None

    def _new_port(self) -> SimulatedAntennaPort:
        return SimulatedAntennaPort()

    def rf_chain_modified(self) -> None:
        """Notify the antenna array that the RF chain configuration of one of its antennas has changed.

        Automatically called when the :attr:`rf_chain<SimulatedAntennaPort.rf_chain>` attribute
        of a :class:`SimulatedAntennaPort` a is modified.
        """

        self.__cached_rf_transmit_map = None
        self.__cached_rf_receive_map = None

    def _rf_transmit_chains(self, default_chain: RfChain) -> Dict[RfChain, List[int]]:
        """Compute a map of all unique RF chains used for transmission.

        Args:

            default_chain:
                The default RF chain to be used if no RF chain is specified for a port.

        Returns:
            A dictionary mapping each unique RF chain to the indices of the ports using it.
        """

        # If the RF chain map has already been computed and the array hasn't changed since then,
        # return the cached map immediately
        if (
            self.__cached_rf_transmit_map is not None
            and self.__cached_default_rf_chain == default_chain
        ):
            return self.__cached_rf_transmit_map.copy()

        unique_rf_chains: Dict[RfChain, List[int]] = dict()
        port_idx = 0
        for port in self.transmit_ports:
            port_rf_chain = port.rf_chain if port.rf_chain is not None else default_chain

            if port_rf_chain not in unique_rf_chains:
                unique_rf_chains[port_rf_chain] = [port_idx]
            else:
                unique_rf_chains[port_rf_chain].append(port_idx)

            port_idx += 1

        # Update the cached RF chain map to save computation time in the future
        self.__cached_default_rf_chain = default_chain
        self.__cached_rf_transmit_map = unique_rf_chains

        # Return the computed RF chain map
        return unique_rf_chains.copy()

    def _rf_receive_chains(self, default_chain: RfChain) -> Dict[RfChain, List[int]]:
        """Compute a map of all unique RF chains used for reception.

        Args:

            default_chain:
                The default RF chain to be used if no RF chain is specified for a port.

        Returns:
            A dictionary mapping each unique RF chain to the indices of the ports using it.
        """

        # If the RF chain map has already been computed and the array hasn't changed since then,
        # return the cached map immediately
        if (
            self.__cached_rf_receive_map is not None
            and self.__cached_default_rf_chain == default_chain
        ):
            return self.__cached_rf_receive_map.copy()

        unique_rf_chains: Dict[RfChain, List[int]] = dict()
        port_idx = 0
        for port in self.receive_ports:
            port_rf_chain = port.rf_chain if port.rf_chain is not None else default_chain

            if port_rf_chain not in unique_rf_chains:
                unique_rf_chains[port_rf_chain] = [port_idx]
            else:
                unique_rf_chains[port_rf_chain].append(port_idx)

            port_idx += 1

        # Update the cached RF chain map to save computation time in the future
        self.__cached_rf_receive_map = unique_rf_chains

        # Return the computed RF chain map
        return unique_rf_chains.copy()

    @staticmethod
    def __combine_rf_propagations(
        num_streams: int,
        sampling_rate: float,
        carrier_frequency: float,
        propagations: Sequence[Signal],
        stream_indices,
    ) -> Signal:
        """Combine multiple RF chain propagations into a single signal model.

        Args:

            num_streams:
                The number of signal streams to be combined.

            sampling_rate:
                The sampling rate of the signal model to be generated in Hz.

            carrier_frequency:
                The carrier frequency of the signal model to be generated in Hz.

            propagations:
                The RF chain propagations to be combined.

            stream_indices:
                The indices of the signal streams to be combined for each RF chain propagation.

        Returns: The combined signal model.
        """

        # Infer the maximum number of generated samples
        max_num_samples = (
            max(propagations, key=lambda s: s.num_samples).num_samples
            if len(propagations) > 0
            else 0
        )

        # Recombine the propagated signals into a single signal model
        combined_signal = Signal.Create(
            np.zeros((num_streams, max_num_samples), dtype=np.complex128),
            sampling_rate,
            carrier_frequency,
        )
        for propagation, stream_indices in zip(propagations, stream_indices):
            combined_signal[stream_indices, : propagation.num_samples] = propagation.getitem()

        return combined_signal

    def transmit(
        self, signal: Signal, default_rf_chain: RfChain, isolation_model: Isolation | None = None
    ) -> tuple[Signal, Signal]:
        """Transmit a signal over the antenna array.

        The transmission may be distorted by the antennas impulse response / frequency characteristics,
        as well as by the RF chains connected to the array's ports.

        Args:

            signal:
                The signal model to be transmitted.

            default_rf_chain:
                The default RF chain to be used if no RF chain is specified for a port.

            isolation_model:
                Model of the signal leaking from the transmit chains to the receive chains.
                If not specified, no leakage is assumed.

        Returns:
            Tuple of the actually transmitted (distorted) signal model and the leakage signal model.

        Raises:

            ValueError: If the number of signal streams does not match the number of transmit ports.
        """

        if signal.num_streams != self.num_transmit_ports:
            raise ValueError(
                f"Number of signal streams does not match number of transmit ports ({signal.num_streams} != {self.num_transmit_ports})"
            )

        # Collect all RF chains used by the array
        rf_chains = self._rf_transmit_chains(default_rf_chain)

        # Simulate RF chain transmission for all specified RF chains
        rf_signals: List[Signal] = []
        for rf_chain, stream_indices in rf_chains.items():
            stream_signal = signal.getstreams(stream_indices)
            rf_signals.append(rf_chain.transmit(stream_signal))

        # Recombine the RF chain transmissions into a single signal model
        combined_rf_signal = self.__combine_rf_propagations(
            signal.num_streams,
            signal.sampling_rate,
            signal.carrier_frequency,
            rf_signals,
            rf_chains.values(),
        )

        # Simulate the transmit-receive leakage
        if isolation_model is not None:
            leaking_signal = isolation_model.leak(combined_rf_signal)
        else:
            leaking_signal = combined_rf_signal.from_ndarray(
                np.empty((combined_rf_signal.num_streams, 0), dtype=np.complex128)
            )

        # Simulate antenna transmission for all antennas
        antenna_signals = signal.Empty(
            **signal.kwargs,
            num_streams=self.num_transmit_antennas,
            num_samples=combined_rf_signal.num_samples,
        )

        antenna_idx = 0
        for port, stream_idx in zip(self.transmit_ports, range(combined_rf_signal.num_streams)):
            antenna_input = combined_rf_signal.getstreams(stream_idx)
            for antenna in port.antennas:
                antenna_signals[antenna_idx, :] = antenna.transmit(antenna_input).getitem()
                antenna_idx += 1

        return antenna_signals, leaking_signal

    def receive(
        self,
        impinging_signal: Signal,
        default_rf_chain: RfChain,
        leaking_signal: Signal | None = None,
        coupling_model: Coupling | None = None,
    ) -> Signal:
        """Receive a signal over the antenna array.

        Args:

            impinging_signal:
                The signal model iminging onto the antenna array over the air.

            default_rf_chain:
                The default RF chain to be used if no RF chain is specified for a port.

            leaking_signal:
                The signal model leaking from the antenna array's transmit chains.
                If not specified, no leakage is assumed.

            coupling_model:
                The coupling model to be used to simulate mutual coupling between the antenna elements.
                If not specified, no mutual coupling is assumed.

        Returns: The base-band digital signal model after analog-digital conversion.
        """

        if impinging_signal.num_streams != self.num_receive_antennas:
            raise ValueError(
                f"Number of signal streams does not match number of receiving antennas ({impinging_signal.num_streams} != {self.num_receive_antennas})"
            )

        # Simulate antenna reception for all antennas
        antenna_outputs = impinging_signal.Empty(
            num_streams=self.num_receive_antennas,
            num_samples=impinging_signal.num_samples,
            **impinging_signal.kwargs,
        )
        for antenna_idx, (stream_idx, antenna) in enumerate(
            zip(range(impinging_signal.num_streams), self.receive_antennas)
        ):
            antenna_input_signal = impinging_signal.getstreams(stream_idx)
            antenna_outputs[antenna_idx, :] = antenna.receive(antenna_input_signal).getitem()

        # Simulate mutual coupling between receiving antennas
        if coupling_model is not None:
            antenna_outputs = coupling_model.receive(antenna_outputs)

        # Simulate transmit receive leakage
        if leaking_signal is not None:
            if leaking_signal.num_streams != self.num_receive_antennas:
                raise ValueError(
                    f"Number of signal streams does not match number of receiving antennas ({impinging_signal.num_streams} != {self.num_receive_antennas})"
                )

            antenna_outputs.superimpose(leaking_signal)

        # Simulate RF chain reception for all specified RF chains
        rf_chains = self._rf_receive_chains(default_rf_chain)
        rf_signals: List[Signal] = []
        for rf_chain, stream_indices in rf_chains.items():
            stream_signal = antenna_outputs.getstreams(stream_indices)
            rf_signals.append(rf_chain.receive(stream_signal))

        rf_receptions = self.__combine_rf_propagations(
            self.num_receive_ports,
            antenna_outputs.sampling_rate,
            antenna_outputs.carrier_frequency,
            rf_signals,
            rf_chains.values(),
        )
        return rf_receptions

    def analog_digital_conversion(
        self, rf_signal: Signal, default_rf_chain: RfChain, frame_duration: float
    ) -> Signal:
        """Model analog-digital conversion during reception.

        Args:

            rf_signal:
                The signal model received by the antenna array's RF chains.

            default_rf_chain:
                The default RF chain to be used if no RF chain is specified for a port.

            frame_duration:
                The duration of the frame to be modeled in seconds.

        Returns: The base-band digital signal model after analog-digital conversion.
        """

        # Recall RF chain map
        rf_chains = self._rf_receive_chains(default_rf_chain)

        # Model ADC conversion during reception
        quantized_signals: List[Signal] = []
        for rf_chain, stream_indices in rf_chains.items():
            quantized_signal = rf_chain.adc.convert(
                rf_signal.getstreams(stream_indices), frame_duration
            )
            quantized_signals.append(quantized_signal)

        # Recombine the quantized signals into a single signal model
        combined_quantized_signal = self.__combine_rf_propagations(
            quantized_signal.num_streams,
            quantized_signal.sampling_rate,
            quantized_signal.carrier_frequency,
            quantized_signals,
            rf_chains.values(),
        )
        return combined_quantized_signal

    def visualize_far_field_pattern(
        self, signal: Signal, *, title: str | None = None
    ) -> plt.Figure:
        """Visualize a signal radiated by the antenna array in its far-field.

        Returns: The Figure of the visualization.
        """

        # Collect angle candidates
        zenith_angles = np.linspace(0, 0.5 * pi, 31)
        azimuth_angles = np.linspace(-pi, pi, 31)
        zenith_samples, azimuth_samples = np.meshgrid(zenith_angles[1:], azimuth_angles)
        aoi = np.append(
            np.array([azimuth_samples.flatten(), zenith_samples.flatten()]).T,
            np.zeros((1, 2)),
            axis=0,
        )

        far_field_power = np.empty(aoi.shape[0], dtype=np.float64)
        for idx, (azimuth, zenith) in enumerate(aoi):
            phase_response = self.spherical_phase_response(
                signal.carrier_frequency, azimuth, zenith, AntennaMode.TX
            )

            for block in signal:
                far_field_power[idx] += np.linalg.norm(phase_response @ block, 2) ** 2

        axes: Axes3D
        with Executable.style_context():
            figure, axes = plt.subplots(subplot_kw={"projection": "3d"})
            figure.suptitle(
                "Antenna Array Transmitted Signal Spatial Characteristics"
                if title is None
                else title
            )
            self.__visualize_pattern(axes, far_field_power, aoi)

        return figure

    @overload
    def plot_pattern(
        self,
        carrier_frequency: float,
        mode: Literal[AntennaMode.TX] | Literal[AntennaMode.RX],
        beamforming_weights: np.ndarray | None = None,
        *,
        title: str | None = None,
    ) -> plt.Figure:
        """Plot the antenna array's radiation pattern.

        Args:

            carrier_frequency:
                The carrier frequency of the signal to be transmitted / received in Hz.

            mode:
                The antenna mode to be plotted.

            beamforming_weights:
                The beamforming weights to be used for beamforming.
                If not specified, the weights are assumed to be :math:`1+0\\mathrm{j}`.

            title:
                The title of the plot.
                If not specified, a default title is assumed.
        """
        ...  # pragma: no cover

    @overload
    def plot_pattern(
        self,
        carrier_frequency: float,
        beamformer: TransmitBeamformer | ReceiveBeamformer,
        *,
        title: str | None = None,
    ) -> plt.Figure:
        """Plot the antenna array's radiation pattern.

        Args:

            carrier_frequency:
                The carrier frequency of the signal to be transmitted / received in Hz.

            beamformer:
                The beamformer to be used for beamforming.

            title:
                The title of the plot.
                If not specified, a default title is assumed.
        """
        ...  # pragma: no cover

    def plot_pattern(  # type: ignore[misc]
        self,
        carrier_frequency: float,
        arg_0: (
            Literal[AntennaMode.TX]
            | Literal[AntennaMode.RX]
            | TransmitBeamformer
            | ReceiveBeamformer
        ),
        arg_1: np.ndarray | None = None,
        *,
        title: str | None = None,
        slice_list: list[tuple[tuple, tuple]] | None = None,
    ) -> plt.Figure:
        """Plot the antenna array's radiation pattern.

        Args:

            carrier_frequency:
                The carrier frequency of the signal to be transmitted / received in Hz.

            arg_0 :

                mode : The antenna mode to be plotted.

                OR

                beamformer :The beamformer to be used for beamforming.

            arg_1 : beamforming_weights.

                The beamforming weights to be used for beamforming.
                If not specified, the weights are assumed to be :math:`1+0\\mathrm{j}`.
                Not required when arg_0 is a beamformer.

            title:
                The title of the plot.
                If not specified, a default title is assumed.

            slice_list :

                A list that defines the parameterisation of the slicing planes to plot the radiation pattern in 2D.
                Each element of the list is a tuple with two tuple elements.
                The first tuple corrresponds to the starting point (azimuth,zenith) of the slicing circle.
                The second tuple corresponds to the repective slicing directions of azimuth and zenith.
                If slice_list is not specified, 3D pattern is plotted.
        """

        mode_str: str

        if arg_0 == AntennaMode.TX or isinstance(arg_0, TransmitBeamformer):
            mode_str = "Transmit"

        elif arg_0 == AntennaMode.RX or isinstance(arg_0, ReceiveBeamformer):
            mode_str = "Receive"

        else:
            raise ValueError("Unknown antenna mode encountered")

        res_figure: plt.Figure

        if slice_list is None:
            # Collect angle candidates for 3D Plot
            azimuth_angles = np.linspace(-pi, pi, 31)
            zenith_angles = np.linspace(0, 0.5 * pi, 31)
            zenith_samples, azimuth_samples = np.meshgrid(zenith_angles[1:], azimuth_angles)
            aoi_3d = np.append(
                np.array([azimuth_samples.flatten(), zenith_samples.flatten()]).T,
                np.zeros((1, 2)),
                axis=0,
            )
            power = self.calculate_power(
                carrier_frequency=carrier_frequency, arg_0=arg_0, arg_1=arg_1, aoi=aoi_3d
            )
            axes_1: Axes3D
            with Executable.style_context():
                # Radiation Pattern in 3D
                figure_1, axes_1 = plt.subplots(subplot_kw={"projection": "3d"})
                figure_1.suptitle(
                    f"Antenna Array {mode_str} Characteristics" if title is None else title
                )
                self.__visualize_pattern(axes_1, power, aoi_3d)
            res_figure = figure_1

        elif slice_list is not None:
            # Create a list to store the power values for each (slice_start,slice_direction) pairs
            power_list = []

            for slice_start, slice_direction in slice_list:
                # Obtain the Starting point and the directions
                theta_s = slice_start[1]
                phi_s = slice_start[0]
                theta_d = slice_direction[1]
                phi_d = slice_direction[0]

                # Generate the Vectors for the slicing plane
                v1 = np.array(
                    [
                        np.sin(theta_s) * np.cos(phi_s),
                        np.sin(theta_s) * np.sin(phi_s),
                        np.cos(theta_s),
                    ]
                )  # Unit Vector in the direction of r slected as the first vector for the slicing plane

                theta_cap = np.array(
                    [
                        np.cos(theta_s) * np.cos(phi_s),
                        np.cos(theta_s) * np.sin(phi_s),
                        -np.sin(theta_s),
                    ]
                )  # Unit Vector in the direction of inclination (theta)

                phi_cap = np.array(
                    [-np.sin(phi_s), np.cos(phi_s), 0]
                )  # Unit Vector in the direction of azimuth (phi)

                v2 = theta_d * theta_cap + phi_d * phi_cap  # Second vector for the slicing plane

                # Compute the Circle for the Slice
                num_sample_points = 1000
                angles = np.linspace(0, 2 * np.pi, num_sample_points)

                cv1 = np.outer(np.cos(angles), v1)
                cv2 = np.outer(np.sin(angles), v2)

                carthesian_circle_samples = cv1 + cv2  # Circle in Cartesian Coordinates

                # Obtain the angle candidates to calculate the Power.
                zenith_samples = np.arccos(carthesian_circle_samples[:, 2])
                azimuth_samples = np.arctan2(
                    carthesian_circle_samples[:, 1], carthesian_circle_samples[:, 0]
                )

                aoi_circ = np.array([azimuth_samples, zenith_samples]).T

                # Calculate the Power and store in the list
                power_list.append(
                    self.calculate_power(
                        carrier_frequency=carrier_frequency, arg_0=arg_0, arg_1=arg_1, aoi=aoi_circ
                    )
                )

            # Plot the pattern in 2D
            with Executable.style_context():
                # Radiation Pattern in 3D
                figure_2, axes_2 = plt.subplots()
                figure_2.suptitle(
                    f"Antenna Array {mode_str} Characteristics in 2D "
                    if title is None
                    else f"Radiation Pattern in 2D for {title}"
                )
                self.__visualize_pattern_sliced(axes_2, power_list, slice_list)
            res_figure = figure_2

        return res_figure

    def calculate_power(  # type: ignore[misc]
        self,
        carrier_frequency: float,
        arg_0: (
            Literal[AntennaMode.TX]
            | Literal[AntennaMode.RX]
            | TransmitBeamformer
            | ReceiveBeamformer
        ),
        arg_1: np.ndarray | None = None,
        *,
        aoi: np.ndarray,
    ) -> np.ndarray:
        """Calculate antenna array's radiated or received power.

        Args:

            carrier_frequency:
                The carrier frequency of the signal to be transmitted / received in Hz.

            arg_0 :

                mode : The antenna mode of the array (Transmission or Reception).

                OR

                beamformer :The beamformer to be used for beamforming.

            arg_1 : beamforming_weights.

                The beamforming weights to be used for beamforming.
                If not specified, the weights are assumed to be :math:`1+0\\mathrm{j}`.
                Not required when arg_0 is a beamformer.

            aoi :
                An array that contains the angles of interest.
        """
        num_ports: int
        ports: Sequence[AntennaPort]
        antennas: Sequence[SimulatedAntenna]
        num_antenna_attr: str
        antenna_mode: AntennaMode

        if arg_0 == AntennaMode.TX or isinstance(arg_0, TransmitBeamformer):
            num_ports = self.num_transmit_ports
            ports = self.transmit_ports
            antennas = self.transmit_antennas
            num_antenna_attr = "num_transmit_antennas"
            antenna_mode = AntennaMode.TX

        elif arg_0 == AntennaMode.RX or isinstance(arg_0, ReceiveBeamformer):
            num_ports = self.num_receive_ports
            ports = self.receive_ports
            antennas = self.receive_antennas
            num_antenna_attr = "num_receive_antennas"
            antenna_mode = AntennaMode.RX

        else:
            raise ValueError("Unknown antenna mode encountered")

        # Collect sensor array response for each angle candidate
        antenna_responses = np.array(
            [
                self.spherical_phase_response(carrier_frequency, angles[0], angles[1], antenna_mode)
                for angles in aoi
            ],
            dtype=np.complex128,
        )
        antenna_responses *= np.array([[a.weight for a in antennas]], dtype=np.complex128)

        # Collect port responses for each angle candidate
        port_responses = np.zeros((antenna_responses.shape[0], num_ports), dtype=np.complex128)
        antenna_idx = 0
        for port_idx, port in enumerate(ports):
            num_antennas = getattr(port, num_antenna_attr)
            port_responses[:, port_idx] = (
                np.sum(
                    antenna_responses[:, antenna_idx : antenna_idx + num_antennas],
                    axis=1,
                    keepdims=False,
                )
                / num_antennas
            )
            antenna_idx += num_antennas

        if arg_0 == AntennaMode.TX or arg_0 == AntennaMode.RX:
            beamforming_weights = (
                np.ones(num_ports, dtype=np.complex128) if arg_1 is None else arg_1
            )

            if beamforming_weights.shape != (num_ports,):
                raise ValueError(
                    f"Beamforming weights must have shape ({num_ports},) but have shape {beamforming_weights.shape}"
                )

            power = (
                np.abs(np.sum(port_responses * beamforming_weights, axis=1, keepdims=False)) ** 2
            )

        elif isinstance(arg_0, TransmitBeamformer):
            base_transform = Transformation.From_Translation(np.zeros(3))
            port_responses = arg_0.encode_streams(
                Signal.Create(
                    np.ones(
                        (arg_0.num_transmit_input_streams(self.num_transmit_ports), 1),
                        dtype=np.complex128,
                    ),
                    1.0,
                    carrier_frequency,
                ),
                self.num_transmit_ports,
                TransmitState(
                    0.0,
                    base_transform,
                    np.zeros(3),
                    carrier_frequency,
                    1.0,
                    self.state(base_transform),
                    self.num_transmit_ports,
                ),
            ).getitem()

            antenna_weights = np.empty(self.num_transmit_antennas, dtype=np.complex128)
            antenna_idx = 0
            for response, port in zip(port_responses, self.transmit_ports):
                antenna_weights[antenna_idx : antenna_idx + port.num_transmit_antennas] = response
                antenna_idx += port.num_transmit_antennas

            power = np.abs(antenna_responses @ antenna_weights) ** 2

        elif isinstance(arg_0, ReceiveBeamformer):
            base_transform = Transformation.From_Translation(np.zeros(3))
            power = np.empty(port_responses.shape[0], dtype=np.float64)
            for p, port_response in enumerate(port_responses):
                s = Signal.Create(port_response[:, None], 1.0, carrier_frequency)
                s = arg_0.decode_streams(
                    s,
                    arg_0.num_receive_output_streams(self.num_receive_ports),
                    ReceiveState(
                        0.0,
                        base_transform,
                        np.zeros(3),
                        carrier_frequency,
                        1.0,
                        self.state(base_transform),
                        self.num_receive_ports,
                    ),
                )
                power[p] = np.abs(s.getitem()) ** 2

        return power

    @staticmethod
    def __visualize_pattern(axes: Axes3D, power: np.ndarray, aoi: np.ndarray) -> None:
        power /= power.max()  # Normalize for visualization purposes

        # Threshold the lower values for better visualization
        power[power < 0.01] = 0.01

        # Compute surface
        surface = np.array(
            [
                power * np.cos(aoi[:, 0]) * np.sin(aoi[:, 1]),
                power * np.sin(aoi[:, 0]) * np.sin(aoi[:, 1]),
                power * np.cos(aoi[:, 1]),
            ],
            dtype=np.float64,
        )

        triangles = tri.Triangulation(aoi[:, 0], aoi[:, 1])
        cmap = plt.cm.ScalarMappable(norm=colors.Normalize(power.min(), power.max()), cmap="jet")
        axes.plot_trisurf(
            surface[0, :],
            surface[1, :],
            surface[2, :],
            triangles=triangles.triangles,
            cmap=cmap.cmap,
            norm=cmap.norm,
            linewidth=0.0,
        )

        axes.set_xlim((-1, 1))
        axes.set_ylim((-1, 1))
        axes.set_zlim((0, 1))
        axes.set_xlabel("X")
        axes.set_ylabel("Y")

    @staticmethod
    def __visualize_pattern_sliced(
        axes, power_list: list, slice_list: list[tuple[tuple, tuple]]
    ) -> None:

        # Generate angle Candidates for the plot
        angles = np.linspace(0, 2 * pi, power_list[0].shape[0])

        # Plot the pattern
        for (slice_start, slice_direction), power in zip(slice_list, power_list):
            axes.semilogy(
                angles,
                power,
                label=f"Slice start = ({int(slice_start[0]/np.pi) if slice_start[0]%np.pi == 0 else slice_start[0]/np.pi:.2g}π ,{int(slice_start[1]/np.pi) if slice_start[1]%np.pi == 0 else slice_start[1]/np.pi:.2g}π ),Slice Direction = {slice_direction}",
            )
        axes.set_xlabel("Angles (radians)")
        axes.set_ylabel("Power")
        axes.grid(color="grey")
        axes.legend(loc="upper right")

        # Set X-axis ticks to show multiples of π
        axes.xaxis.set_major_locator(ticker.MultipleLocator(base=np.pi / 2))  # π/2 increments
        axes.xaxis.set_minor_locator(ticker.MultipleLocator(base=np.pi / 6))  # π/6 increments

        # Format tick labels as π fractions
        axes.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x/np.pi) if x%np.pi == 0 else x/np.pi:.2g}π")
        )

    def antenna_state(self, base_pose: Transformation) -> AntennaArrayState:
        """Return the antenna array's state with respect to a base pose.

        Args:

            base_pose (Transformation):
                The base pose to be used as reference.

        Returns: The antenna array's state with respect to the base pose.
        """


class SimulatedUniformArray(
    SimulatedAntennaArray, UniformArray[SimulatedAntennaPort, SimulatedAntenna]
):
    """A uniform array of simulated antennas."""

    def __init__(
        self,
        element: Type[SimulatedAntenna] | SimulatedAntenna,
        spacing: float,
        dimensions: Sequence[int],
        pose: Transformation | None = None,
    ) -> None:
        """
        Args:

            element:
                The anntenna model this uniform array assumes.

            spacing:
                Spacing between the antenna elements in m.

            dimensions:
                The number of antennas in x-, y-, and z-dimension.

            pose:
                The anntena array's transformation with respect to its device.
        """

        # Initialize base classes
        # Not that the order of the base class initialization is important here to presrve the kinematic chain!
        SimulatedAntennaArray.__init__(self, pose)
        UniformArray.__init__(self, element, spacing, dimensions, pose)


class SimulatedCustomArray(
    SimulatedAntennaArray, CustomAntennaArray[SimulatedAntennaPort, SimulatedAntenna]
):
    """A custom array of simulated antennas."""

    def __init__(
        self,
        ports: Sequence[SimulatedAntennaPort | SimulatedAntenna] = None,
        pose: Transformation | None = None,
    ) -> None:
        """
        Args:

            ports:
                Sequence of antenna ports available within this array.
                If antennas are passed instead of ports, the ports are automatically created.
                If not specified, an empty array is assumed.

            pose:
                The anntena array's transformation with respect to its device.
        """

        # Initialize base classes
        CustomAntennaArray.__init__(self, ports, pose)
        SimulatedAntennaArray.__init__(self, pose)

    def add_port(self, port: SimulatedAntennaPort) -> None:
        CustomAntennaArray.add_port(self, port)
        self.rf_chain_modified()

    def remove_port(self, port: SimulatedAntennaPort) -> None:
        CustomAntennaArray.remove_port(self, port)
        self.rf_chain_modified()

    def add_antenna(self, antenna: SimulatedAntenna) -> None:
        CustomAntennaArray.add_antenna(self, antenna)
        self.rf_chain_modified()
