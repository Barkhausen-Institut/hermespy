# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

from .channel import Channel, ChannelRealization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class IdealChannelRealization(ChannelRealization):
    """Realization of an ideal channel."""

    ...  # pragma: no cover


class IdealChannel(Channel[IdealChannelRealization]):
    """An ideal distortion-less channel.

    It also serves as a base class for all other channel models.

    For MIMO systems, the received signal is the addition of the signal transmitted at all
    antennas.
    The channel will provide `number_rx_antennas` outputs to a signal
    consisting of `number_tx_antennas` inputs. Depending on the channel model,
    a random number generator, given by `rnd` may be needed. The sampling rate is
    the same at both input and output of the channel, and is given by `sampling_rate`
    samples/second.
    """

    yaml_tag: str = "Channel"

    def realize(self, num_samples: int, _: float) -> IdealChannelRealization:
        """Generate a new channel impulse response.

        Note that this is the core routine from which :meth:`Channel.propagate` will create the channel state.

        Args:

            num_samples (int):
                Number of samples :math:`N` within the impulse response.

        Returns:

                Numpy aray representing the impulse response for all propagation paths between antennas.
                4-dimensional tensor of size :math:`M_\\mathrm{Rx} \\times M_\\mathrm{Tx} \\times N \\times (L+1)` where
                :math:`M_\\mathrm{Rx}` is the number of receiving antennas,
                :math:`M_\\mathrm{Tx}` is the number of transmitting antennas,
                :math:`N` is the number of propagated samples and
                :math:`L` is the maximum path delay (in samples).
                For the ideal  channel in the base class :math:`L = 0`.
        """

        if self.transmitter is None or self.receiver is None:
            raise RuntimeError("Channel is floating, making impulse response simulation impossible")

        # MISO case
        if self.receiver.antennas.num_antennas == 1:
            spatial_response = np.ones((1, self.transmitter.antennas.num_antennas), dtype=complex)

        # SIMO case
        elif self.transmitter.antennas.num_antennas == 1:
            spatial_response = np.ones((self.receiver.antennas.num_antennas, 1), dtype=complex)

        # MIMO case
        else:
            spatial_response = np.eye(self.receiver.antennas.num_antennas, self.transmitter.antennas.num_antennas, dtype=complex)

        # Scale by channel gain and add dimension for delay response
        impulse_response = np.sqrt(self.gain) * np.expand_dims(np.repeat(spatial_response[:, :, np.newaxis], num_samples, 2), axis=3)

        # Save newly generated response as most recent impulse response
        self.recent_response = impulse_response

        # Return resulting impulse response
        return IdealChannelRealization(self, impulse_response)
