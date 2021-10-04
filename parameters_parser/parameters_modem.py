import configparser
import ast
from abc import ABC, abstractmethod
import os
from typing import List, Any

import numpy as np

from parameters_parser.parameters_crc_encoder import ParametersCrcEncoder
from parameters_parser.parameters_repetition_encoder import ParametersRepetitionEncoder
from parameters_parser.parameters_ldpc_encoder import ParametersLdpcEncoder
from parameters_parser.parameters_block_interleaver import ParametersBlockInterleaver
from parameters_parser.parameters_rf_chain import ParametersRfChain
from parameters_parser.parameters_encoder import ParametersEncoder


class ParametersModem(ABC):
    """This abstract class implements the parser of the transceiver parameters.

    Attributes:
        technology_val list(str): valid technology values
        technology (ParametersWaveformGenerator): object containing all technology-specific parameters
        position (list(float)): 3D-location of transceiver (in meters)
        velocity (list(float)): 3D- velocity of transceiver (in m/s)
        number_of_antennas (int): number of tx/rx antennas
        carrier_frequency (float): transceiver carrier frequency (in Hz)
                supported_encoders List(str): list of valid encoders
        dir_encoding_parameters(str): path where we can find the encoding parametersr
        encoding_type(List[str]): type of encoders
        device_type (str): defines which device type the modem ist
        antenna_spacing (float): Ratio between antenna distance and lambda/2
    """

    technology_val = ["PSK_QAM", "CHIRP_FSK", "OFDM"]
    supported_encoders = ["REPETITION", "LDPC"]
    device_type_val = ["BASE_STATION", "UE"]

    def __init__(self) -> None:
        """creates a parsing object, that will manage the transceiver parameters."""
        self.technology: Any
        self.position: List[float] = []
        self.velocity: List[float] = []
        self.number_of_antennas = 1
        self.carrier_frequency: float = 0.
        self.tx_power = 0.
        self.dir_encoding_parameters = os.path.join(os.getcwd(), '_settings', 'coding')
        self.rf_chain_parameters = os.path.join(os.getcwd(), '_settings')
        self.encoding_type = [""]
        self.encoding_params = []
        self.block_interleaver_m = 1
        self.block_interleaver_n = 1
        self._encoder_param_file = "NONE"
        self._encoded_bits_n = 1
        self._data_bits_k = 1

        self._rf_chain_param_file = ""
        self.rf_chain = ParametersRfChain()

        self.device_type = ""
        self.cov_matrix = np.array([])
        self.antenna_spacing = 1.
        self.crc_bits = 0

    @abstractmethod
    def read_params(self, section: configparser.SectionProxy) -> None:
        """reads the channel parameters contained in the section 'section' of a given configuration file."""
        self.position = ast.literal_eval(section.get("position", fallback="[0,0,0]"))
        self.velocity = ast.literal_eval(section.get("velocity", fallback="[0,0,0]"))
        self.number_of_antennas = section.getint("number_of_antennas", fallback=1)
        self.device_type = section.get("device_type", fallback="UE").upper()
        self.antenna_spacing = section.getfloat("antenna_spacing", fallback=1.)
        self.block_interleaver_m = section.getint("block_interleaver_m", fallback=1)
        self.block_interleaver_n = section.getint("block_interleaver_n", fallback=1)

        self.crc_bits = section.getint("crc_bits", fallback=0)
        tx_power_db = section.getfloat("tx_power_db", fallback=0.)
        if tx_power_db == 0:
            self.tx_power = 0.
        else:
            self.tx_power = 10 ** (tx_power_db / 10.)

        self._encoder_param_file = section.get("encoder_param_file", fallback=self._encoder_param_file)
        self._rf_chain_param_file = section.get("rf_chain_param_file", fallback=None)

    @abstractmethod
    def check_params(self, param_path: str = "") -> None:
        """checks the validity of the parameters."""

        if not self._encoder_param_file.upper() == "NONE":
            if not os.path.exists(os.path.join(param_path, "coding")):
                print(f"Directory {os.path.join(param_path, 'coding')} does not exist.")
                print("Taking default settings directory.")
            else:
                self.dir_encoding_parameters = os.path.join(param_path, "coding")

        if (not isinstance(self.position, list) or not len(self.position) == 3 or
                not all(isinstance(x, (int, float)) for x in self.position)):
            raise ValueError(
                'position (' +
                ' '.join(
                    str(e) for e in self.position) +
                ' must be a 3-D number vector')

        if (not isinstance(self.velocity, list) or not len(self.velocity) == 3 or
                not all(isinstance(x, (int, float)) for x in self.velocity)):
            raise ValueError(
                'velocity (' +
                ' '.join(
                    str(e) for e in self.velocity) +
                ' must be a 3-D number vector')

        self.velocity = np.asarray(self.velocity)

        if self.device_type not in self.device_type_val:
            raise ValueError(f"Device type {self.device_type} not supported")

        if not isinstance(self.number_of_antennas,
                          int) or self.number_of_antennas < 1:
            raise ValueError('number_of_antennas (' +
                             str(self.number_of_antennas) +
                             'must be an integer > 0')

        if self.antenna_spacing <= 0:
            raise ValueError('antenna spacing must be > 0.')


        # read encoder parameters file
        if self._encoder_param_file.upper() == "NONE":
            self.encoding_params.append(ParametersRepetitionEncoder())
            self.encoding_params[-1].encoded_bits_n = 1
            self.encoding_params[-1].data_bits_k = 1
        else:
            encoding_params_file_path = os.path.join(
                self.dir_encoding_parameters, self._encoder_param_file)
            self._read_encoding_file(encoding_params_file_path)

        self.encoding_params.append(
            ParametersBlockInterleaver(
                self.block_interleaver_m, self.block_interleaver_n))
        self.encoding_type.append("BLOCK_INTERLEAVER")


        params_crc = ParametersCrcEncoder(
            self.crc_bits,
            self._get_minimum_data_bits_k(self.encoding_params)
        )
        self.encoding_params.append(params_crc)

        self.encoding_type.append("CRC_BITS")

        if self._rf_chain_param_file:
            if self._rf_chain_param_file.upper() == "NONE":
                self.rf_chain = None
            else:
                self.rf_chain.read_params(os.path.join(param_path, self._rf_chain_param_file))
        else:
            self.rf_chain = None

    def _get_minimum_data_bits_k(self, encoding_params: List[ParametersEncoder]) -> int:
        p = min(
            self.encoding_params,
            key=lambda params: params.data_bits_k)

        return p.data_bits_k

    def _read_encoding_file(self, encoding_params_file_path: str) -> None:
        config = configparser.ConfigParser()
        if not os.path.exists(encoding_params_file_path):
            raise ValueError(f'File {encoding_params_file_path} does not exist.')

        config.read(encoding_params_file_path)
        self.encoding_type.append(config["General"].get("type").upper())

        if self.encoding_type[-1] not in self.supported_encoders:
            raise ValueError(f"Encoding Type {self.encoding_type} not supported")

        encoding_parameters: ParametersEncoder
        if self.encoding_type[-1] == "REPETITION":
            encoding_parameters = ParametersRepetitionEncoder()
        elif self.encoding_type[-1] == "LDPC":
            encoding_parameters = ParametersLdpcEncoder()

        self._encoded_bits_n = config["General"].getint("encoded_bits_n", fallback=1)
        self._data_bits_k = config["General"].getint("data_bits_k", fallback=1)

        encoding_parameters.encoded_bits_n = self._encoded_bits_n
        encoding_parameters.data_bits_k = self._data_bits_k

        encoding_parameters.read_params(config["General"])
        self.encoding_params.append(encoding_parameters)

    def _calculate_correlation_matrices(self):
        a = 0.
        if self._correlation == "CUSTOM":
            a = self._custom_correlation
        elif self.device_type == "BASE_STATION":
            a = self.alpha_val[self._correlation]
        elif self.device_type == "UE":
            a = self.beta_val[self._correlation]

        if self._correlation == "CUSTOM":
            self.cov_matrix = np.eye(self.number_of_antennas, self.number_of_antennas)
            for i in range(self.number_of_antennas):
                for j in range(self.number_of_antennas):
                    self.cov_matrix[i, j] = a ** ((abs(i-j)) * self.antenna_spacing)
        else:
            if self.number_of_antennas == 1:
                self.cov_matrix = np.ones((1, 1))
            elif self.number_of_antennas == 2:
                self.cov_matrix = np.array([[1, a],
                                            [a, 1]])
            elif (self.number_of_antennas & (self.number_of_antennas - 1)) == 1:
                raise ValueError("number of antennas must be power of 2")
            elif self.number_of_antennas >= 4:
                self.cov_matrix = np.array([[1, a ** (1 / 9), a ** (4 / 9), a],
                                            [a**(1 / 9), 1, a**(1 / 9), a**(4 / 9)],
                                            [a**(4 / 9), a**(1 / 9), 1, a**(1 / 9)],
                                            [a, a**(4 / 9), a**(1 / 9), 1]])
