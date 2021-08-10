from typing import List
import configparser

from parameters_parser.parameters_modem import ParametersModem
from parameters_parser.parameters_tx_modem import ParametersTxModem
from parameters_parser.parameters_ofdm import ParametersOfdm


class ParametersRxModem(ParametersModem):
    """This class implements the parser of the receiver parameters."""

    def __init__(self) -> None:
        """creates a parsing object, that will manage the receiver parameters."""
        super().__init__()

        self.tx_modem = 0
        self.id = 0

    def read_params(self, section: configparser.SectionProxy) -> None:
        """reads the modem parameters contained in the section 'section' of a given configuration file."""
        super().read_params(section)

        self.tx_modem = section.getint("tx_modem") - 1
        self.track_length = section.getint("track_length", fallback=1)
        self.track_angle = section.getint("track_angle", fallback=1)

    def check_params(
            self, tx_modem_params: List[ParametersTxModem] = None, param_path: str = "") -> None:
        """validates all the modem simulation parameters"""
        try:
            super().check_params(param_path)
        except ValueError as error_details:
            raise error_details

        if self.tx_modem < 0 or self.tx_modem >= len(tx_modem_params):
            raise ValueError(
                'tx modem (' + str(self.tx_modem + 1) + ') was not defined')

        self.technology = tx_modem_params[self.tx_modem].technology

        if isinstance(self.technology, ParametersOfdm):
            self.technology.number_rx_antennas = self.number_of_antennas

        self.carrier_frequency = tx_modem_params[self.tx_modem].carrier_frequency
        self.encoding_type = tx_modem_params[self.tx_modem].encoding_type
        self.encoding_params = tx_modem_params[self.tx_modem].encoding_params
        self.crc_bits = tx_modem_params[self.tx_modem].crc_bits
