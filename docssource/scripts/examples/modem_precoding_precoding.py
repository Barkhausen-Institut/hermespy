# -*- coding: utf-8 -*-

from __future__ import annotations

from hermespy.modem import (
    TransmitSymbolCoding,
    ReceiveSymbolCoding,
    DFT as TransmitSymbolEncoder,
    DFT as ReceiveSymbolDecoder,
    SimplexLink as Modem,
)


# Initialize a new transmit precoding configuration
tx_precoding = TransmitSymbolCoding()
rx_preocoding = ReceiveSymbolCoding()

# Add precoding steps
tx_precoding[0] = TransmitSymbolEncoder()
rx_preocoding[0] = ReceiveSymbolDecoder()

# Initialize a new modem
modem = Modem()

# Add precoding configuration to modem
modem.transmit_symbol_coding = tx_precoding
modem.receive_symbol_coding = rx_preocoding

# Directly configure the precoding configuration
modem.transmit_symbol_coding[0] = TransmitSymbolEncoder()
modem.receive_symbol_coding[1] = ReceiveSymbolDecoder()
