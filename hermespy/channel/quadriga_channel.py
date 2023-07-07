# -*- coding: utf-8 -*-
"""
======================
Quadriga Channel Model
======================
"""

from __future__ import annotations
from typing import Optional

import numpy as np

from hermespy.channel import Channel, ChannelRealization, QuadrigaInterface

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class QuadrigaChannel(Channel):
    """Quadriga Channel Model.

    Maps the output of the QuadrigaInterface to fit into hermes software architecture.
    """

    yaml_tag = "Quadriga"
    serialized_attributes = Channel.serialized_attributes.union({"active", "gain"})

    # Reference to the interface class
    __interface: Optional[QuadrigaInterface]

    def __init__(self, *args, interface: Optional[QuadrigaInterface] = None, **kwargs) -> None:
        """
        Args:

            interface (Optional[QuadrigaInterface], optional):
                Specifies the consisdered Quadriga interface.
                Defaults to None.
        """

        # Init base channel class
        Channel.__init__(self, *args, **kwargs)

        # Save interface settings
        self.__interface = interface

        # Register this channel at the interface
        self.__quadriga_interface.register_channel(self)

    def __del__(self) -> None:
        """Quadriga channel object destructor.

        Automatically un-registers channel objects at the interface.
        """

        self.__quadriga_interface.unregister_channel(self)

    @property
    def __quadriga_interface(self) -> QuadrigaInterface:
        """Access global Quadriga interface as property.

        Returns:
            QuadrigaInterface: Global Quadriga interface.
        """

        return QuadrigaInterface.GlobalInstance() if self.__interface is None else self.__interface  # type: ignore

    def realize(self, num_samples: int, sampling_rate: float) -> ChannelRealization:
        # Query the quadriga interface for a new impulse response
        path_gains, path_delays = self.__quadriga_interface.get_impulse_response(self)

        max_delay_in_samples = np.around(np.max(path_delays) * sampling_rate).astype(int)

        impulse_response = np.zeros((self.receiver.num_antennas, self.transmitter.num_antennas, num_samples, max_delay_in_samples + 1), dtype=complex)

        for tx_antenna in range(self.transmitter.num_antennas):
            for rx_antenna in range(self.receiver.num_antennas):
                # of dimension, #paths x #snap_shots, along the third dimension are the samples
                # choose first snapshot, i.e. assume static
                cir_txa_rxa = path_gains[rx_antenna, tx_antenna, :]
                tau_txa_rxa = path_delays[rx_antenna, tx_antenna, :]

                time_delay_in_samples_vec = np.around(tau_txa_rxa * sampling_rate).astype(int)

                for delay_idx, delay_in_samples in enumerate(time_delay_in_samples_vec):
                    impulse_response[rx_antenna, tx_antenna, :, delay_in_samples] += cir_txa_rxa[delay_idx]

        return ChannelRealization(self, np.sqrt(self.gain) * impulse_response)
