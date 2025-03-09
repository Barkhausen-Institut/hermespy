# -*- coding: utf-8 -*-

from __future__ import annotations

from hermespy.modem import BaseModem, CommunicationWaveform
from ..noise import NoiseLevel

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class CommunicationNoiseLevel(NoiseLevel):
    """Base class for all communication noise level configuration classes."""

    __reference: BaseModem | CommunicationWaveform

    def __init__(
        self, reference: BaseModem | CommunicationWaveform, level: float = float("inf")
    ) -> None:
        """
        Args:
            reference (BaseModem | CommunicationWaveform):
                Reference with respect to which the noise level is defined.

            level (float, optional): Noise level relative to the reference'
        """

        # Init base class
        NoiseLevel.__init__(self)

        # Init class attributes
        self.level = level
        self.__reference = reference

    @property
    def level(self) -> float:
        """Communication relative noise level.

        Raises:

            ValueError: For non-positive noise levels.
        """

        return self.__level

    @level.setter
    def level(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Communication noise level must be positive")

        self.__level = value

    @property
    def reference(self) -> BaseModem | CommunicationWaveform:
        """Reference of the noise level.

        Returns: Reference of the noise level.
        """

        return self.__reference

    @reference.setter
    def reference(self, value: BaseModem | CommunicationWaveform) -> None:
        self.__reference = value

    def _get_reference_waveform(self) -> CommunicationWaveform:
        """Waveform of the reference signal.

        Returns: Waveform of the reference signal.

        Raises:

            RuntimeError: If a modem withiout a waveform is used as a reference.
        """

        if isinstance(self.reference, CommunicationWaveform):
            return self.reference
        else:
            if self.reference.waveform is None:
                raise RuntimeError(
                    "The reference modem has no waveform configured. Noise level cannot be determined."
                )
            return self.reference.waveform


class EBN0(CommunicationNoiseLevel):
    """Fixed noise power configuration."""

    def get_power(self) -> float:
        return self._get_reference_waveform().bit_energy / self.level

    @property
    def title(self) -> str:
        return "E_B/N_0"


class ESN0(CommunicationNoiseLevel):
    """Fixed noise power configuration."""

    def get_power(self) -> float:
        return self._get_reference_waveform().symbol_energy / self.level

    @property
    def title(self) -> str:
        return "E_S/N_0"
