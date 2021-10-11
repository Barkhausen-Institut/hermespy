# -*- coding: utf-8 -*-
"""HermesPy simulation configuration.
"""

from __future__ import annotations

from .executable import Executable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Simulation(Executable):
    """HermesPy simulation configuration."""

    yaml_tag = u'Simulation'

    def __init__(self) -> None:
        """Object initialization."""

        Executable.__init__(self)

    def run(self) -> None:
        """Run the full simulation configuration."""

        # Iterate over scenarios
        for scenario in self.scenarios:

            # Generate data bits to be transmitted
            data_bits = scenario.generate_data_bits()

            # Generate radio-frequency band signal emitted from each transmitter
            transmitted_signals = scenario.transmit(data_bits=data_bits)

            # Simulate propagation over channel models
            propagated_signals = scenario.propagate(transmitted_signals)

            # Receive and demodulate signal
            received_bits = scenario.receive(propagated_signals)
