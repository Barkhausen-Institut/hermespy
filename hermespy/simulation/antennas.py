# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import overload, Sequence, Type, Literal
from typing_extensions import override

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.tri as tri
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
from scipy.constants import pi

from hermespy.beamforming import TransmitBeamformer, ReceiveBeamformer
from hermespy.core import (
    Antenna,
    AntennaArray,
    AntennaMode,
    CustomAntennaArray,
    Dipole,
    Executable,
    IdealAntenna,
    LinearAntenna,
    PatchAntenna,
    ReceiveState,
    Signal,
    SignalBlock,
    Transformation,
    TransmitState,
    UniformArray,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SimulatedAntenna(Antenna):
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
            signal: The signal model to be transmitted.

        Returns: The actually transmitted (distorted) signal model.

        Raises:
            ValueError: If the signal has more than one stream.
        """

        if signal.num_streams > 1:
            raise ValueError("Only single-streamed signal can be transmitted over a single antenna")

        return Signal.Create(
            [self.weight * b for b in signal.blocks],  # type: ignore
            sampling_rate=signal.sampling_rate,
            carrier_frequency=signal.carrier_frequency,
            noise_power=signal.noise_power,
            delay=signal.delay,
            offsets=signal.block_offsets,
        )

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

        if signal.num_streams > 1:
            raise ValueError("Only single-streamed signal can be received over a single antenna")

        return Signal.Create(
            [(self.weight * b).view(SignalBlock) for b in signal.blocks],
            sampling_rate=signal.sampling_rate,
            carrier_frequency=signal.carrier_frequency,
            noise_power=signal.noise_power,
            delay=signal.delay,
            offsets=signal.block_offsets,
        )


class SimulatedDipole(SimulatedAntenna, Dipole):
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


class SimulatedIdealAntenna(SimulatedAntenna, IdealAntenna):
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

    @override
    def copy(self) -> SimulatedIdealAntenna:
        return SimulatedIdealAntenna(self.mode, self.pose, self.weight)


class SimulatedLinearAntenna(SimulatedAntenna, LinearAntenna):
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

    @override
    def copy(self) -> SimulatedLinearAntenna:
        return SimulatedLinearAntenna(self.mode, self.slant, self.pose, self.weight)


class SimulatedPatchAntenna(SimulatedAntenna, PatchAntenna):
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

    @override
    def copy(self) -> SimulatedPatchAntenna:
        return SimulatedPatchAntenna(self.mode, self.pose, self.weight)


class SimulatedAntennaArray(AntennaArray[SimulatedAntenna]):
    """Array of simulated antennas."""

    def transmit(self, signal: Signal) -> Signal:
        """Transmit a signal over the antenna array.

        The transmission may be distorted by the antennas impulse response / frequency characteristics,
        as well as by the RF chains connected to the array's ports.

        Args:
            signal: The signal model to be transmitted.

        Returns:
            The signal model emerging from the antenna array's transmit antennas.

        Raises:
            ValueError: If the number of signal streams does not match the number of transmit ports.
        """

        if signal.num_streams != self.num_transmit_antennas:
            raise ValueError(
                f"Number of signal streams does not match number of transmit antennas ({signal.num_streams} != {self.num_transmit_antennas})"
            )

        # Simulate antenna transmission for all antennas
        antenna_signals = signal.copy()
        for idx, (stream, antenna) in enumerate(zip(signal, self.transmit_antennas)):
            antenna_signals[idx, :] = antenna.transmit(stream)

        return antenna_signals

    def receive(self, signal: Signal) -> Signal:
        """Receive a signal over the antenna array.

        Args:
            signal: The signal impinging onto the antenna array.

        Returns:
            The signal model emerging from the antenna ports after reception.
        """

        if signal.num_streams != self.num_receive_antennas:
            raise ValueError(
                f"Number of signal streams does not match number of receiving antennas ({signal.num_streams} != {self.num_receive_antennas})"
            )

        # Simulate antenna reception for all antennas
        antenna_outputs = signal.copy()
        for idx, (stream, antenna) in enumerate(zip(signal, self.receive_antennas)):
            antenna_outputs[idx, :] = antenna.receive(stream)

        return antenna_outputs

    def visualize_far_field_pattern(self, signal: Signal, *, title: str | None = None) -> Figure:
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

            for block in signal.blocks:
                far_field_power[idx] += (
                    np.linalg.norm(phase_response @ block.view(np.ndarray), 2) ** 2
                )

        axes: Axes3D
        with Executable.style_context():
            figure, axes = plt.subplots(subplot_kw={"projection": "3d"})  # type: ignore
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
    ) -> Figure:
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
    ) -> Figure:
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
    ) -> Figure:
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

        res_figure: Figure

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
                figure_1, axes_1 = plt.subplots(subplot_kw={"projection": "3d"})  # type: ignore
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

                mode: The antenna mode of the array (Transmission or Reception).

                OR

                beamformer: The beamformer to be used for beamforming.

            arg_1 : beamforming_weights.

                The beamforming weights to be used for beamforming.
                If not specified, the weights are assumed to be :math:`1+0\\mathrm{j}`.
                Not required when arg_0 is a beamformer.

            aoi :
                An array that contains the angles of interest.
        """

        antennas: Sequence[SimulatedAntenna]
        antenna_mode: AntennaMode

        if arg_0 == AntennaMode.TX or isinstance(arg_0, TransmitBeamformer):
            antennas = self.transmit_antennas
            antenna_mode = AntennaMode.TX

        elif arg_0 == AntennaMode.RX or isinstance(arg_0, ReceiveBeamformer):
            antennas = self.receive_antennas
            antenna_mode = AntennaMode.RX

        else:
            raise ValueError("Unknown antenna mode encountered")

        # Collect sensor array response for each angle candidate
        array_responses = np.array(
            [
                self.spherical_phase_response(carrier_frequency, angles[0], angles[1], antenna_mode)
                for angles in aoi
            ],
            dtype=np.complex128,
        )
        array_responses *= np.array([[a.weight for a in antennas]], dtype=np.complex128)

        if arg_0 == AntennaMode.TX or arg_0 == AntennaMode.RX:
            beamforming_weights = (
                np.ones(array_responses.shape[1], dtype=np.complex128) if arg_1 is None else arg_1
            )

            power = (
                np.abs(np.sum(array_responses * beamforming_weights, axis=1, keepdims=False)) ** 2
            )

        elif isinstance(arg_0, TransmitBeamformer):
            base_transform = Transformation.From_Translation(np.zeros(3))
            beamforming_weights = arg_0.encode_streams(
                Signal.Create(np.ones((1, 1), dtype=np.complex128), 1.0, carrier_frequency),
                self.num_transmit_antennas,
                TransmitState(
                    0.0,
                    base_transform,
                    np.zeros(3),
                    carrier_frequency,
                    1.0,
                    1,
                    self.state(base_transform),
                    self.num_transmit_antennas,
                    self.num_transmit_antennas,
                ),
            ).view(np.ndarray)

            power = np.abs(array_responses @ beamforming_weights) ** 2

        elif isinstance(arg_0, ReceiveBeamformer):
            base_transform = Transformation.From_Translation(np.zeros(3))
            power = np.empty(array_responses.shape[0], dtype=np.float64)
            for a, array_response in enumerate(array_responses):
                s = Signal.Create(array_response.reshape((-1, 1)), 1.0, carrier_frequency)
                s = arg_0.decode_streams(
                    s,
                    1,
                    ReceiveState(
                        0.0,
                        base_transform,
                        np.zeros(3),
                        carrier_frequency,
                        1.0,
                        1,
                        self.state(base_transform),
                        1,
                        1,
                    ),
                )
                power[a] = s.power

        return power

    @staticmethod
    def __visualize_pattern(axes: Axes3D, power: np.ndarray, aoi: np.ndarray) -> None:
        power = power.flatten()
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
            antialiased=True,
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
                label=f"Slice start = ({int(slice_start[0]/np.pi) if slice_start[0] % np.pi == 0 else slice_start[0]/np.pi:.2g}π ,{int(slice_start[1]/np.pi) if slice_start[1] % np.pi == 0 else slice_start[1]/np.pi:.2g}π ),Slice Direction = {slice_direction}",
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
            ticker.FuncFormatter(lambda x, _: f"{int(x/np.pi) if x % np.pi == 0 else x/np.pi:.2g}π")
        )


class SimulatedUniformArray(SimulatedAntennaArray, UniformArray[SimulatedAntenna]):
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


class SimulatedCustomArray(SimulatedAntennaArray, CustomAntennaArray[SimulatedAntenna]):
    """A custom array of simulated antennas."""

    def __init__(
        self, ports: Sequence[SimulatedAntenna] | None = None, pose: Transformation | None = None
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
