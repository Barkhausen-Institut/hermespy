import configparser
import os
from typing import Any

from parameters_parser.parameters_modem import ParametersModem
from parameters_parser.parameters_psk_qam import ParametersPskQam
from parameters_parser.parameters_chirp_fsk import ParametersChirpFsk
from parameters_parser.parameters_ofdm import ParametersOfdm


class ParametersTxModem(ParametersModem):
    """This class implements the parser of the transmitter parameters."""

    supported_power_amplifier_models = ["NONE", "CLIP", "RAPP", "SALEH", "CUSTOM"]

    def __init__(self) -> None:
        super().__init__()
        self.id = 0
        self.crc_bits = 1

        self._technology_param_file = ""

    def read_params(self, section: configparser.SectionProxy) -> None:
        """ This method reads and checks all the parameters from a given section in a parameter file

        Args:
            section (configparser.SectionProxy): section containing the modem parameters.
        """
        super().read_params(section)

        self._technology_param_file = section.get("technology_param_file")
        self.carrier_frequency = section.getfloat("carrier_frequency")
        self.crc_bits = section.getint("crc_bits", 1)

    def check_params(self, param_path: str = "") -> None:
        try:
            super().check_params(param_path)
        except ValueError as error_details:
            raise error_details

        if self.carrier_frequency < 0:
            raise ValueError(
                'carrier_frequency (' + str(self.carrier_frequency) + 'must be >= 0')
        if self.crc_bits < 0:
            raise ValueError(f"Number of crc_bits must be positive, currently it is {self.crc_bits}.")

        # read technology-specific parameters
        config = configparser.ConfigParser()
        filename = os.path.join(param_path, self._technology_param_file)

        if not os.path.exists(filename):
            raise ValueError('technology file (' + filename + 'does not exist')

        config.read(filename)
        technology = config["General"].get("technology").upper()

        if technology.upper() not in ParametersModem.technology_val:
            raise ValueError(
                'invalid technology (' +
                technology +
                ') in file ' +
                filename)

        tech_parameters: Any
        if technology == "PSK_QAM":
            tech_parameters = ParametersPskQam()
        elif technology == "CHIRP_FSK":
            tech_parameters = ParametersChirpFsk()
        elif technology == "OFDM":
            tech_parameters = ParametersOfdm(number_tx_antennas=self.number_of_antennas)
        else:
            raise ValueError("invalid technology")
        
        tech_parameters.read_params(filename)
        self.technology = tech_parameters
