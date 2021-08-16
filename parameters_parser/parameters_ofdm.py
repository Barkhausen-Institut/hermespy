import configparser
from enum import Enum
from collections import namedtuple
from typing import List
import re
from dataclasses import dataclass, field
from abc import ABC

import numpy as np

from parameters_parser.parameters_waveform_generator import ParametersWaveformGenerator

FrameElementDef = namedtuple('FrameElementDef', 'type number_of_samples')


class ResourceType(Enum):
    REFERENCE = 1
    DATA = 2
    NULL = 3


MultipleRes = namedtuple('MultipleRes', ['ResourceType', 'number'])
ResourcePattern = namedtuple(
    'ResourcePattern',
    ['MultipleRes', 'number'],
    defaults=[[], 0])


@dataclass
class FrameElement(ABC):
    no_samples: int = 0

@dataclass
class GuardInterval(FrameElement):
    pass


@dataclass
class OfdmSymbolConfig(FrameElement):
    cyclic_prefix_samples: int = 0
    resource_types: List[ResourcePattern] = field(default_factory=list)


class ParametersOfdm(ParametersWaveformGenerator):
    """This class implements the parameters parser of an OFDM modem

    Description of attributes can be found in the documentation.

    Attributes:
        subcarrier_spacing(float):
        fft_size(int):
        number_occupied_subcarriers(int):
        cp_ratio(numpy.ndarray):
        precoding(str):
        precoding_val(List[str]): list of the valid precoding types
        modulation_order:
        modulation_order_val(List[int]):
        oversampling_factor(int):
        dc_suppression:

        channel_estimation(str):
        equalization(str):

        mimo_scheme(str):
        number_tx_antennas(int):
        number_rx_antennas(int):
        number_streams(int): relevant for MIMO

        frame_guard_interval(float):
        frame_structure(List[FrameElement]):
        ofdm_symbol_resources_mapping(List[List[MultipleRes]]):
        ofdm_symbol_configs(List[OfdmSymbolConfig]): each ofdm symbol has a certain config.


        bits_in_frame:
    """

    modulation_order_val = [2, 4, 16, 64, 256]
    precoding_val = ['NONE', 'DFT']
    mimo_val = ['SFBC', 'SM', 'SM-ZF', 'SM-MMSE', 'MRC', 'SC', 'NONE']
    channel_estimation_val = [
        'IDEAL',
        'IDEAL_PREAMBLE',
        'IDEAL_MIDAMBLE',
        'IDEAL_POSTAMBLE']
    equalization_val = ['ZF', 'MMSE']

    def __init__(self, number_tx_antennas: int = 1, number_rx_antennas: int = 1) -> None:
        """creates a parsing object, that will manage the transceiver parameters.
        """
        super().__init__()
        ####################
        # input parameters

        # Modulation parameters
        self.subcarrier_spacing = 0.
        self.fft_size = 1
        self.number_occupied_subcarriers = 1
        self.cp_ratio = np.array([])
        self.precoding = ""
        self.modulation_order = 2
        self.oversampling_factor = 1
        self.dc_suppression = True

        # receiver parameters
        self.channel_estimation = ""
        self.equalization = ""

        # mimo parameters
        self.mimo_scheme = ""
        self.number_tx_antennas = number_tx_antennas
        self.number_rx_antennas = number_rx_antennas
        self.number_streams = 1

        # Frame parameters
        self.frame_guard_interval = 0.
        self.frame_structure: List[FrameElement] = []
        self.ofdm_symbol_resources_mapping: List[List[ResourcePattern]] = []
        self.ofdm_symbol_configs: List[OfdmSymbolConfig] = []
        self.reference_symbols = np.array([1.0])

        ######################
        # derived parameters
        self.bits_in_frame = 0

    def read_params(self, file_name: str) -> None:
        """reads the modem parameters

        Args:
            file_name(str): name/path of configuration file containing the parameters
        """

        super().read_params(file_name)

        config = configparser.ConfigParser()
        config.read(file_name)

        cfg = config['Modulation']

        self.subcarrier_spacing = cfg.getfloat('subcarrier_spacing')
        self.fft_size = cfg.getint('fft_size')
        self.oversampling_factor = cfg.getint("oversampling_factor")
        self.number_occupied_subcarriers = cfg.getint('number_occupied_subcarriers')
        self.cp_ratio = cfg.get('cp_ratio')
        self.cp_ratio = np.fromstring(self.cp_ratio, sep=',')
        self.modulation_order = cfg.getint("modulation_order")
        self.precoding = cfg.get('precoding')
        self.dc_suppression = cfg.getboolean('dc_suppression', fallback=self.dc_suppression)

        cfg = config['Receiver']

        self.channel_estimation = cfg.get('channel_estimation')
        self.equalization = cfg.get('equalization')

        cfg = config['MIMO']

        self.mimo_scheme = cfg.get('mimo_scheme').upper()

        cfg = config['Frame']
        self.frame_guard_interval = cfg.getfloat("frame_guard_interval")
        self._frame_structure_str = cfg.get("frame_structure")
        self._ofdm_symbol_res_str = cfg.get("ofdm_symbol_resources")
        self._cp_lengths_str = cfg.get("cp_length")

        ref_symbols = cfg.get("reference_symbols", fallback="1")
        self.reference_symbols = np.fromstring(ref_symbols, dtype=complex, sep=',')

        self._check_params()

    def _check_params(self) -> None:
        """checks the validity of the parameters and calculates derived parameters

        Raises:
            ValueError: if there is an invalid value in the parameters
        """

        top_header = 'ERROR reading OFDM modem parameters'

        #######################
        # check modulation parameters
        msg_header = top_header + ', Section "Modulation", '
        self.sampling_rate = self.subcarrier_spacing * self.fft_size * self.oversampling_factor

        if self.sampling_rate <= 0:
            raise ValueError(
                msg_header +
                f'sampling rate ({self.sampling_rate}) must be positive')

        if self.number_occupied_subcarriers > self.fft_size:
            raise ValueError(msg_header + (f'number_occupied_subcarriers({self.number_occupied_subcarriers})'
                                           f' cannot be larger than fft_size({self.fft_size})'))

        if self.modulation_order not in ParametersOfdm.modulation_order_val:
            raise ValueError(
                msg_header +
                f'modulation order ({self.modulation_order}) not supported')

        self.precoding = self.precoding.upper()
        if self.precoding not in ParametersOfdm.precoding_val:
            raise ValueError(
                msg_header +
                f'precoding ({self.precoding}) not supported')

        #############################
        # check receiver parameters
        msg_header = top_header + ', Section "Receiver", '
        self.channel_estimation = self.channel_estimation.upper()
        if self.channel_estimation not in ParametersOfdm.channel_estimation_val:
            raise ValueError(
                msg_header +
                f'channel_estimation ({self.channel_estimation}) not supported')

        if self.equalization not in ParametersOfdm.equalization_val:
            raise ValueError(
                msg_header +
                f'equalization ({self.equalization}) not supported')

        #############################
        # check MIMO parameters
        msg_header = top_header + ', Section "MIMO", '
        if self.mimo_scheme not in ParametersOfdm.mimo_val:
            raise ValueError(
                msg_header +
                f'MIMO scheme ({self.mimo_scheme}) not supported')

        if self.mimo_scheme == 'SM':
            self.mimo_scheme = 'SM-ZF'

        if self.mimo_scheme.find('SM') == 0:
            # no MIMO precoding is supported yet
            self.number_streams = self.number_tx_antennas
        else:
            self.number_streams = 1

        #############################
        # check frame parameters
        msg_header = top_header + ', Section "Frame", '

        self.ofdm_symbol_resources_mapping = self.read_ofdm_symbol_resources(self._ofdm_symbol_res_str)
        self.ofdm_symbol_configs = self.map_resource_types_to_symbols(self.ofdm_symbol_resources_mapping)
        self.read_cp_lengths(self._cp_lengths_str, self.ofdm_symbol_configs)

        self.frame_structure = self.read_frame_structure(self._frame_structure_str)

        self._calculate_bits_in_frame()
        self._check_no_ofdm_symbol_resources(self.ofdm_symbol_resources_mapping)

    def _check_no_ofdm_symbol_resources(self, res: List[List[ResourcePattern]]) -> None:
        for ofdm_symbol_res in res:
            no_res = 0
            for res_pattern in ofdm_symbol_res:
                for pattern_el_idx in range(res_pattern.number):
                    for res in res_pattern.MultipleRes:
                        no_res += res.number  # type: ignore

            if no_res != self.number_occupied_subcarriers:
                raise ValueError(f"We got {no_res} ressources but we expect {self.number_occupied_subcarriers}")

    def map_resource_types_to_symbols(self, res: List[List[ResourcePattern]]) -> List[OfdmSymbolConfig]:
        """Assigns the resource patterns to a ofdm symbol config each"""
        ofdm_symbol_configs: List[OfdmSymbolConfig] = []
        for ofdm_symbol_res in res:
            ofdm_symbol = OfdmSymbolConfig(
                resource_types=ofdm_symbol_res,
                no_samples=self.fft_size
            )
            ofdm_symbol_configs.append(ofdm_symbol)

        return ofdm_symbol_configs

    def read_cp_lengths(self, cp_lengths_str: str, ofdm_symbol_configs: List[OfdmSymbolConfig]):
        cp_lengths_str = cp_lengths_str.replace(' ', '').lower()
        cp_symbols_split_up = cp_lengths_str.split(',')

        if not (len(ofdm_symbol_configs) == len(cp_symbols_split_up)):
            raise ValueError("There must be one CP defined for each ofdm symbol.")

        for ofdm_symbol_idx, cp_symbol in enumerate(cp_symbols_split_up):
            if not (cp_symbol[0] == 'c'):
                raise ValueError("Unsupported cp symbol. It must start with a c")
            no_samples = 0
            if len(cp_symbol) > 1:
                if not (cp_symbol[1].isdigit()):
                    raise ValueError(f"{cp_symbol[1]} must be a number.")

                cp_ratio_idx = int(cp_symbol[1])
                if cp_ratio_idx > len(self.cp_ratio):
                    raise ValueError("Too large CP ratio idx.")
            else:
                raise ValueError("Please indicate index of cp ratio to choose from.")
            cp_ratio_idx -= 1
            no_samples = int(
                np.around(
                    self.cp_ratio[cp_ratio_idx]*self.fft_size)
            )

            ofdm_symbol_configs[ofdm_symbol_idx].cyclic_prefix_samples = no_samples

    def read_ofdm_symbol_resources(self, ofdm_symbol_resources_str: str) -> List[List[ResourcePattern]]:
        ofdm_symbol_resources_str = ofdm_symbol_resources_str.replace(' ', '')
        ofdm_symbol_resources_str = ofdm_symbol_resources_str.lower()
        res: List[List[ResourcePattern]] = []

        regex_not_permitted = '[a-ce-mo-qs-z]'

        neg_replicate_pattern = re.compile(regex_not_permitted, flags=re.IGNORECASE)

        if neg_replicate_pattern.search(ofdm_symbol_resources_str) is not None:
            raise ValueError("There was an error in the symbol structure.")

        ofdm_symbol_resources_split: List[str] = ofdm_symbol_resources_str.split(',')

        if len(ofdm_symbol_resources_split) > 10:
            raise ValueError("Number of ofdm symbol resouces must not be larger than 10")

        for idx, ofdm_symbol_str in enumerate(ofdm_symbol_resources_split):

            res.append([])
            # get number of blocks
            no_blocks = 1
            starts_with_repetition = re.match('[0-9]+', ofdm_symbol_str)
            if starts_with_repetition is not None:
                ofdm_symbol_str = re.sub('^[0-9]+', '', ofdm_symbol_str)
                no_blocks = int(starts_with_repetition[0])

                if not (ofdm_symbol_str.startswith('(') and ofdm_symbol_str.endswith(')')):
                    raise ValueError("Parenthesis required.")
                else:
                    ofdm_symbol_str = ofdm_symbol_str[1:-1]
            pattern = ResourcePattern(
                MultipleRes=self._read_resource_pattern(ofdm_symbol_str),
                number=no_blocks
            )
            res[idx].append(pattern)

        return res

    def _read_resource_pattern(self, resource_pattern_str: str) -> List[MultipleRes]:
        pattern: List[MultipleRes] = []
        regex_permitted = '[0-9]*[rdn]'
        replicate_pattern = re.compile(regex_permitted, flags=re.IGNORECASE)

        for resource in replicate_pattern.findall(resource_pattern_str):
            number_match = re.match('[0-9]+', resource)
            if number_match is None:
                no_resources = 1
            else:
                no_resources = int(number_match.group(0))

            if resource[-1] == 'r':
                resource_type = ResourceType.REFERENCE
            elif resource[-1] == 'n':
                resource_type = ResourceType.NULL
            elif resource[-1] == 'd':
                resource_type = ResourceType.DATA
            else:
                raise ValueError(f"Ressource type {resource[-1]} not supported")

            pattern.append(MultipleRes(resource_type, no_resources))
        return pattern

    def read_frame_structure(self, frame_structure: str) -> List[FrameElement]:
        frame_structure = frame_structure.replace(' ', '')
        frame_structure = frame_structure.lower()
        repetititon_pattern = re.compile('[0-9]+\(')
        number_pattern = re.compile('[0-9]+')

        frame_config: List[FrameElement] = []

        for frame_block_str in frame_structure.split(','):
            # get number of blocks
            repetitions = 1
            starts_with_repetition = repetititon_pattern.match(frame_block_str)
            number_found = number_pattern.match(frame_block_str)

            if starts_with_repetition is not None:
                frame_block_str = re.sub('^[0-9]+', '', frame_block_str)
                repetitions = int(number_found[0])

                if not (frame_block_str.startswith('(') and frame_block_str.endswith(')')):
                    raise ValueError("Error reading OFDM frame structure, parenthesis required.")
                else:
                    frame_block_str = frame_block_str[1:-1]
            frame_config.extend(
                repetitions * self._parse_frame_block(frame_block_str))
        return frame_config

    def _parse_frame_block(self, frame_block_str: str) -> List[FrameElement]:
        symbols: List[FrameElement] = []

        for symbol in frame_block_str:
            if symbol == 'g':
                guard_interval = GuardInterval()
                guard_interval.no_samples = int(
                    np.around(
                        self.frame_guard_interval
                        * self.subcarrier_spacing
                        * self.fft_size)
                )
                symbols.append(guard_interval)
            else:
                if not symbol.isdigit():
                    raise ValueError("Must be a number.")
                ofdm_symbol_idx = int(symbol)
                if ofdm_symbol_idx > len(self.ofdm_symbol_configs):
                    raise ValueError("No corresponding ofdm Symbol found.")
                symbols.append(self.ofdm_symbol_configs[ofdm_symbol_idx - 1])
        return symbols

    def _calculate_bits_in_frame(self):
        for element in self.frame_structure:
            if isinstance(element, OfdmSymbolConfig):
                for res_pattern in element.resource_types:
                    for res in res_pattern.MultipleRes:
                        if res.ResourceType == ResourceType.DATA:
                            self.bits_in_frame += (
                                res.number
                                * int(np.log2(self.modulation_order))
                                * res_pattern.number
                                * self.number_streams
                            )
