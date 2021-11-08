from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from beamformer import Beamformer, TransmissionDirection

if TYPE_CHECKING:
    from modem import Modem


class ConventionalBeamformer(Beamformer):

    __focused_modem: Modem

    def __init__(self, modem: Modem, focused_modem: Modem = None):
        """Class initialization.

        Args:
            modem (Modem):
                Modem instance this beamformer is linked to.

            focused_modem (Modem, optional):
                Modem towards which the beamformer is focusing its power pattern.
        """

        Beamformer.__init__(self, modem)
        self.__focused_modem = focused_modem

    @property
    def focused_modem(self) -> Modem:
        return self.__focused_modem

    def weights(self, direction: TransmissionDirection, azimuth: float, elevation: float) -> np.array:
        """Compute the beamforming weights towards a desired direction.

        Args:
            direction (TransmissionDirection):
                The direction of transmission, i.e. receive or transmit mode.

            azimuth (float):
                The considered azimuth angle in radians.

            elevation (float):
                The considered elevation angle in radians.

        Returns:
            np.array:
                A vector containing the computed beamforming weights.
        """

        wave_vector = self.wave_vector(azimuth, elevation)

        if direction == TransmissionDirection.Rx:
            wave_vector *= -1

        return np.array([np.exp(1j * wave_vector @ p) for p in self.modem.topology], dtype=complex)

    @property
    def num_streams(self) -> int:
        """The number of channels available streams after beamforming.

        Returns (int):
            The number of available streams.
        """

        return 1
