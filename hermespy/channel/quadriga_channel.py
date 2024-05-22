# -*- coding: utf-8 -*-

from __future__ import annotations
from math import ceil
from typing import Type


import numpy as np
from h5py import Group

from hermespy.core import (
    ChannelStateInformation,
    ChannelStateFormat,
    Device,
    HDFSerializable,
    Signal,
)
from hermespy.channel import Channel, ChannelRealization, InterpolationMode, QuadrigaInterface

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class QuadrigaChannelRealization(ChannelRealization):
    """Realization of a Quadriga channel.

    Generated by the :meth:`realize<QuadrigaChannel.realize>` method of :class:`QuadrigaChannel`.
    """

    __path_gains: np.ndarray
    __path_delays: np.ndarray

    def __init__(
        self,
        alpha_device: Device,
        beta_device: Device,
        gain: float,
        path_gains: np.ndarray,
        path_delays: np.ndarray,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
    ) -> None:
        """
        Args:

            alpha_device (Device):
                First device linked by the realized channel.

            beta_device (Device):
                Second device linked by the realized channel.

            gain (float):
                Channel gain.

            path_gains (np.ndarray):
                Path gains.

            path_delays (np.ndarray):
                Path delays.

            interpolation_mode (InterpolationMode, optional):
                Interpolation mode.
                Defaults to InterpolationMode.NEAREST.
        """

        # Initialize base class
        ChannelRealization.__init__(self, alpha_device, beta_device, gain, interpolation_mode)

        # Initialize class attributes
        self.__path_gains = path_gains
        self.__path_delays = path_delays

    @property
    def path_gains(self) -> np.ndarray:
        """Path gains."""

        return self.__path_gains

    @property
    def path_delays(self) -> np.ndarray:
        """Path delays."""

        return self.__path_delays

    def _propagate(
        self,
        signal: Signal,
        transmitter: Device,
        receiver: Device,
        interpolation: InterpolationMode,
    ) -> Signal:
        if signal.num_samples > self.path_gains.shape[3]:
            raise ValueError(
                f"Quadriga channel realization does not support signals with more samples than the channel realization ({signal.num_samples} > {self.path_gains.shape[2]})."
            )

        max_delay_in_samples = ceil(np.max(self.path_delays) * signal.sampling_rate)
        propagated_samples = np.zeros(
            (receiver.antennas.num_receive_antennas, signal.num_samples + max_delay_in_samples),
            dtype=np.complex_,
        )

        for tx_antenna in range(transmitter.antennas.num_transmit_antennas):
            for rx_antenna in range(receiver.antennas.num_receive_antennas):
                # of dimension, #paths x #snap_shots, along the third dimension are the samples
                # choose first snapshot, i.e. assume static
                cir_txa_rxa = self.__path_gains[rx_antenna, tx_antenna, ::]
                tau_txa_rxa = self.__path_delays[rx_antenna, tx_antenna, ::]

                time_delay_in_samples_vec = np.around(tau_txa_rxa * signal.sampling_rate).astype(
                    int
                )

                for delay_idx, delay_in_samples in enumerate(time_delay_in_samples_vec):
                    propagated_samples[
                        rx_antenna, delay_in_samples : delay_in_samples + signal.num_samples
                    ] += (
                        cir_txa_rxa[delay_idx, : signal.num_samples] * signal[tx_antenna, :].flatten()
                    )

        return signal.from_ndarray(propagated_samples)

    def state(
        self,
        transmitter: Device,
        receiver: Device,
        delay: float,
        sampling_rate: float,
        num_samples: int,
        max_num_taps: int,
    ) -> ChannelStateInformation:
        max_delay_in_samples = ceil(np.max(self.path_delays) * sampling_rate)
        num_taps = min(max_num_taps, max_delay_in_samples + 1)

        impulse_response = np.zeros(
            (
                receiver.antennas.num_receive_antennas,
                transmitter.antennas.num_transmit_antennas,
                num_samples,
                num_taps,
            ),
            dtype=np.complex_,
        )

        for tx_antenna in range(receiver.antennas.num_receive_antennas):
            for rx_antenna in range(transmitter.antennas.num_transmit_antennas):
                # of dimension, #paths x #snap_shots, along the third dimension are the samples
                # choose first snapshot, i.e. assume static
                cir_txa_rxa = self.path_gains[rx_antenna, tx_antenna, ::]
                tau_txa_rxa = self.path_delays[rx_antenna, tx_antenna, :]

                time_delay_in_samples_vec = np.around(tau_txa_rxa * sampling_rate).astype(int)
                time_delay_in_samples_vec = time_delay_in_samples_vec[
                    time_delay_in_samples_vec < num_taps
                ]

                for delay_idx, delay_in_samples in enumerate(time_delay_in_samples_vec):
                    impulse_response[rx_antenna, tx_antenna, :, delay_in_samples] += cir_txa_rxa[
                        delay_idx
                    ]

        impulse_response *= np.sqrt(self.gain)
        return ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, impulse_response)

    def to_HDF(self, group: Group) -> None:
        ChannelRealization.to_HDF(self, group)
        HDFSerializable._write_dataset(group, "path_gains", self.path_gains)
        HDFSerializable._write_dataset(group, "path_delays", self.path_delays)

    @classmethod
    def From_HDF(
        cls: Type[QuadrigaChannelRealization],
        group: Group,
        alpha_device: Device,
        beta_device: Device,
    ) -> QuadrigaChannelRealization:
        params = cls._parameters_from_HDF(group)
        params["path_gains"] = np.array(group["path_gains"], dtype=np.complex_)
        params["path_delays"] = np.array(group["path_delays"], dtype=np.float_)

        return cls(alpha_device, beta_device, **params)


class QuadrigaChannel(Channel):
    """Quadriga Channel Model.

    Maps the output of the :class:`QuadrigaInterface<hermespy.channel.quadriga_interface.QuadrigaInterface>` to fit into Hermes' software architecture.
    """

    yaml_tag = "Quadriga"

    __interface: QuadrigaInterface | None  # Reference to the interface class

    def __init__(self, *args, interface: QuadrigaInterface | None = None, **kwargs) -> None:
        """
        Args:

            interface (QuadrigaInterface, optional):
                Specifies the consisdered Quadriga interface.
                Defaults to None.
        """

        # Init base channel class
        Channel.__init__(self, *args, **kwargs)

        # Save interface settings
        self.__interface = interface

        # Register this channel at the interface
        self._quadriga_interface.register_channel(self)

    def __del__(self) -> None:
        """Quadriga channel object destructor.

        Automatically un-registers channel objects at the interface.
        """

        self._quadriga_interface.unregister_channel(self)

    @property
    def _quadriga_interface(self) -> QuadrigaInterface:
        """Access global Quadriga interface as property.

        Returns:
            QuadrigaInterface: Global Quadriga interface.
        """

        return QuadrigaInterface.GlobalInstance() if self.__interface is None else self.__interface  # type: ignore

    def _realize(self) -> ChannelRealization:
        # Query the quadriga interface for a new impulse response
        (path_gains, path_delays) = self._quadriga_interface.get_impulse_response(self)

        # Return the channel realization
        return QuadrigaChannelRealization(
            self.alpha_device,
            self.beta_device,
            self.gain,
            path_gains,
            path_delays,
            self.interpolation_mode,
        )

    def recall_realization(self, group: Group) -> QuadrigaChannelRealization:
        return QuadrigaChannelRealization.From_HDF(group, self.alpha_device, self.beta_device)
