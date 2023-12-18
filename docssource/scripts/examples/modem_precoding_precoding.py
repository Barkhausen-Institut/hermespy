# -*- coding: utf-8 -*-

from __future__ import annotations

from hermespy.modem import (
    SymbolPrecoding,
    Alamouti as SymbolPrecoder,
    ReceivingModem as Modem,
)


# Initialize a new precoding configuration
precoding = SymbolPrecoding()

# Add precoding steps
precoding[0] = SymbolPrecoder()
precoding[1] = SymbolPrecoder()

# Initialize a new modem
modem = Modem()

# Add precoding configuration to modem
modem.precoding = precoding

# Directly configure the precoding configuration
modem.precoding[0] = SymbolPrecoder()
modem.precoding[1] = SymbolPrecoder()
