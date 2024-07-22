# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Set, Any
from h5py import Group
from math import ceil

import numpy as np

from sionna import rt  # type: ignore

from hermespy.channel.channel import ChannelSampleHook, InterpolationMode, LinkState
from hermespy.core import ChannelStateInformation, ChannelStateFormat, SignalBlock

from .channel import Channel, ChannelRealization, ChannelSample

__author__ = "Egor Achkasov"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Egor Achkasov"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SionnaRTChannelSample(ChannelSample):
    """Sample of a Sionna RT channel realization.

    Generated by the :meth:`_sample<SionnaRTChannelRealization._sample>` routine of :class:`SionnaRTChannelRealization`.
    """

    paths: rt.Paths

    # Channel states and delays before Doppler effect application.
    # These members are Tensorflow tensors and SHOULD NOT be used directly.
    # It is a backup of sionna.rt.Paths properties used to reverse Doppler effect.
    __a: Any
    __tau: Any

    __gain: float

    def __init__(self, paths: rt.Paths, gain: float, state: LinkState) -> None:
        """
        Args:

            paths (sionna.rt.Paths):
                Ray-Tracing paths in this sample. Should be generated with _realize of Realization.

            gain (float):
                Linear channel power factor.

            state (ChannelState):
                State of the channel at the time of sampling.
        """

        # Initialize base class
        ChannelSample.__init__(self, state)

        # Initialize class attributes
        self.paths = paths
        self.__a = rt.tf.identity(paths._a)
        self.__tau = rt.tf.identity(paths._tau)
        self.__gain = gain

    @property
    def expected_energy_scale(self) -> float:
        """Expected linear scaling of a propagated signal's energy at each receiving antenna.

        Required to compute the expected energy of a signal after propagation,
        and therfore signal-to-noise ratios (SNRs) and signal-to-interference-plus-noise ratios (SINRs).

        TODO Current implementation is technically incorrect.
        """

        return np.abs(np.sum(self.__a))

    def __apply_doppler(self, num_samples: int) -> tuple:
        """Apply Doppler effect on the paths (via sionna.rt.Paths.apply_doppler).
        Cast and reshape the channel state (a and tau) as assumed in self.state and self._propagate.
        Set the original _a and _tau of self.paths back to reverse the doppler effect.
        This workflow allows reusing of the same Paths object for different propagations.

        Returns:
            a (np.ndarray): gains. Shape (num_rx_ants, num_tx_ants, num_paths, num_samples)
            tau (np.ndarray): delays. Shape (num_rx_ants, num_tx_ants, num_paths)
        """
        # Apply doppler
        self.paths.apply_doppler(sampling_frequency=self.bandwidth,
                                 num_time_steps=num_samples,
                                 tx_velocities=self.transmitter_velocity,
                                 rx_velocities=self.receiver_velocity)

        # Get and cast CIR
        a, tau = self.paths.cir()
        a = a.numpy()[0, 0, :, 0, :, :, :]
        tau = tau.numpy()[0, 0, 0, :]

        # Restore paths to the original state
        self.paths._a = rt.tf.identity(self.__a)
        self.paths._tau = rt.tf.identity(self.__tau)

        return a, tau

    def state(
        self,
        num_samples: int,
        max_num_taps: int,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST
    ) -> ChannelStateInformation:
        # Apply Doppler effect and get the channel impulse response
        a, tau = self.__apply_doppler(num_samples)

        # Init result
        max_delay = np.max(tau) if tau.size != 0 else 0
        max_delay_in_samples = min(max_num_taps, ceil(max_delay * self.bandwidth))
        raw_state = np.zeros((
            self.num_receive_antennas,
            self.num_transmit_antennas,
            num_samples,
            1 + max_delay_in_samples), dtype=np.complex_)
        # If no paths hit the target, then return an empty state
        if a.size == 0 or tau.size == 0:
            return ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, raw_state)

        for a_p, tau_p in zip(np.moveaxis(a, -2, 0), np.moveaxis(tau, -1, 0)):
            if tau_p < 0:
                continue
            delay_tap_index = int(tau_p * self.bandwidth)
            if delay_tap_index >= max_num_taps:
                continue  # pragma: no cover

            raw_state[:, :, :, delay_tap_index] += a_p

        raw_state *= np.sqrt(self.__gain)
        return ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, raw_state)

    def _propagate(
        self,
        signal_block: SignalBlock,
        interpolation: InterpolationMode,
    ) -> SignalBlock:
        # Calculate the resulting signal block parameters
        sr_ratio = self.receiver_state.sampling_rate / self.transmitter_state.sampling_rate
        offset_new = int(signal_block.offset * sr_ratio)
        num_streams_new = self.num_receive_antennas
        num_samples_new = int(signal_block.num_samples * sr_ratio)

        # Apply Doppler effect and get the channel impulse response
        a, tau = self.__apply_doppler(signal_block.num_samples)
        # If no paths hit the target, then return a zeroed signal
        if a.size == 0 or tau.size == 0:
            return SignalBlock(np.zeros((num_streams_new, num_samples_new),
                                        signal_block.dtype),
                               offset_new)

        # Set other attributes
        max_delay = np.max(tau)
        max_delay_in_samples = ceil(max_delay * self.bandwidth)
        propagated_samples = np.zeros(
            (num_streams_new, signal_block.num_samples + max_delay_in_samples),
            dtype=signal_block.dtype
        )

        # Prepare the optimal einsum path ahead of time for faster execution
        einsum_subscripts = "ijk,jk->ik"
        einsum_path = np.einsum_path(
            einsum_subscripts, a[:, :, 0, :], signal_block, optimize="optimal"
        )[0]

        # For each path
        for a_p, tau_p in zip(np.moveaxis(a, -2, 0), np.moveaxis(tau, -1, 0)):
            if tau_p == -1.0:
                continue
            t = int(tau_p * self.bandwidth)
            propagated_samples[
                :, t : t + signal_block.num_samples
            ] += np.einsum(einsum_subscripts, a_p, signal_block, optimize=einsum_path)

        propagated_samples *= np.sqrt(self.__gain)
        return SignalBlock(propagated_samples, offset_new)


class SionnaRTChannelRealization(ChannelRealization[SionnaRTChannelSample]):
    """Realization of a Sionna RT channel.

    Generated by the :meth:`_realize()<SionnaRTChannel._realize>` routine of :class:`SionnaRTChannels<SionnaRTChannel>`.
    """

    scene: rt.Scene

    def __init__(self,
                 scene: rt.Scene,
                 sample_hooks: Set[ChannelSampleHook] | None = None,
                 gain: float = 1.) -> None:
        super().__init__(sample_hooks, gain)
        self.scene = scene

    def _sample(self, state: LinkState) -> SionnaRTChannelSample:
        # Clear the scene
        self.scene._transmitters.clear()
        self.scene._receivers.clear()
        self.scene._tx_array = None
        self.scene._rx_array = None

        # init self.scene.tx_array
        tx_antenna = rt.Antenna("iso", "V")
        tx_positions = [
            a.position
            for a in state.transmitter.antennas.transmit_antennas
        ]
        self.scene.tx_array = rt.AntennaArray(tx_antenna, tx_positions)

        # init self.scene.rx_array
        rx_antenna = rt.Antenna("iso", "V")
        rx_positions = [
            a.position
            for a in state.receiver.antennas.receive_antennas
        ]
        self.scene.rx_array = rt.AntennaArray(rx_antenna, rx_positions)

        # init tx and rx
        self.scene.add(rt.Transmitter("Alpha device", state.transmitter.position))
        self.scene.add(rt.Receiver("Beta device", state.receiver.position))

        # set other self.scene params
        self.scene.frequency = state.transmitter.carrier_frequency
        self.scene.synthetic_array = True

        # calculate paths
        paths = self.scene.compute_paths()
        paths.normalize_delays = False

        # construct the sample
        return SionnaRTChannelSample(paths, self.gain, state)

    def _reciprocal_sample(
        self, sample: SionnaRTChannelSample, state: LinkState
    ) -> SionnaRTChannelSample:
        return self._sample(state)

    def to_HDF(self, group: Group) -> None:
        group.attrs["gain"] = self.gain

    @staticmethod
    def From_HDF(
        scene: rt.Scene,
        group: Group,
        sample_hooks: Set[ChannelSampleHook[SionnaRTChannelSample]]
    ) -> SionnaRTChannelRealization:
        return SionnaRTChannelRealization(scene, sample_hooks, group.attrs["gain"])


class SionnaRTChannel(Channel[SionnaRTChannelRealization, SionnaRTChannelSample]):
    """Sionna ray-tracing channel.

    Refer to :doc:`/api/channel/sionna-rt` for further information.
    """

    yaml_tag: str = "SionnaRT"

    scene: rt.Scene

    def __init__(self, scene: rt.Scene, gain: float = 1, seed: int | None = None) -> None:
        super().__init__(gain, seed)
        self.scene = scene

    def _realize(self) -> SionnaRTChannelRealization:
        return SionnaRTChannelRealization(self.scene, self.sample_hooks, self.gain)

    def recall_realization(self, group: Group) -> SionnaRTChannelRealization:
        return SionnaRTChannelRealization.From_HDF(self.scene, group, self.sample_hooks)
