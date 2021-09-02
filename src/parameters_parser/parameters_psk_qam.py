import configparser
import warnings
import math

import numpy as np
from parameters_parser.parameters_waveform_generator import ParametersWaveformGenerator


class ParametersPskQam(ParametersWaveformGenerator):
    """This class implements the parser of a PSK/QAM modem

    Attributes:
        modulation_order (int): constellation size
        modulation_is_complex (bool): True for complex modulation (PSK/QAM), False for real modulation (PAM)
        symbol_rate (float): modulation symbol rate (in bauds)
        oversampling_factor (int): number of samples per symbol.
                                   The (minimum) sampling rate will be 'symbol_rate * oversampling_factor'
        filter_type (str): type of tx shaping pulse. It can be one of the values defined in 'filter_type_val'.
        roll_off_factor (float): roll-off factor between 0 and 1. This is only relevant if 'filter_type' is either
                                 'RAISED_COSINE' and 'ROOT_RAISED_COSINE'
        filter_length_in_symbols (int): if 'filter_type' is either 'RAISED_COSINE' and 'ROOT_RAISED_COSINE', then the
                                        filter impulse response will have a length of 'filter_length_in_symbols'
                                        modulation symbols.
                                        if 'fliter_type' is 'FMCW', then it corresponds to the chirp duration
        bandwidth(float): filter bandwidth

        number_preamble_symbols:
        number_data_symbols:
        number_postamble_symbols: the transceiver will frames containing 'number_preamble_symbols' unit-valued reference
                                  symbols at the beginning, followed by 'number_data_symbols' modulated symbols and
                                  'number_postamble_symbols' at the end.
        pilot_symbol_rate: symbol rate (in bauds) of the pilot symbols
        guard_interval: frames are followed by 'guard_interval' seconds of silence.
        bits_per_symbol:
        bits_in_frame:
    """

    modulation_order_complex_val = [2, 4, 8, 16, 64, 256]
    modulation_order_real_val = [2, 4, 8, 16]
    filter_type_val = [
        "ROOT_RAISED_COSINE",
        "RAISED_COSINE",
        "RECTANGULAR",
        "FMCW",
        "NONE"]
    equalizer_val = ["NONE", "ZF", "MMSE"]

    def __init__(self) -> None:
        """
        creates a parsing object, that will manage the transceiver parameters.
        """
        super().__init__()

        # Modulation parameters
        self.modulation_order = 0
        self.modulation_is_complex = False
        self.symbol_rate = 0.
        self.filter_type = ""
        self.roll_off_factor = 0.
        self.oversampling_factor = 0
        self.filter_length_in_symbols = 0.
        self.bandwidth = 0.
        self.chirp_duration = 0.
        self.chirp_bandwidth = 0.
        self.pulse_width = 0.

        # Receiver parameters
        self.equalizer = ""

        # Frame parameters
        self.number_preamble_symbols = 0
        self.number_data_symbols = 0
        self.number_postamble_symbols = 0
        self.pilot_symbol_rate = 0.
        self.guard_interval = 0.
        self.bits_in_frame = 0
        self.bits_per_symbol = 0

        # misc
        self._chirp_duration_str = ""
        self._chirp_bandwidth_str = ""
        self._pilot_symbol_rate_str = ""

    def read_params(self, file_name: str) -> None:
        """ reads the modem parameters contained in the configuration file
            'file_name'."""
        super().read_params(file_name)

        config = configparser.ConfigParser()
        config.read(file_name)

        cfg = config['Modulation']
        self.modulation_order = cfg.getint("modulation_order")
        self.modulation_is_complex = cfg.getboolean("is_complex")
        self.symbol_rate = cfg.getfloat("symbol_rate")
        self.filter_type = cfg.get("filter_type").upper()
        self.oversampling_factor = cfg.getint(
            "oversampling_factor", fallback=2)

        self.roll_off_factor = cfg.getfloat("roll_off_factor", fallback=.3)
        self.filter_length_in_symbols = cfg.getfloat(
            "filter_length", fallback=16)

        self._chirp_duration_str = cfg.get(
            "chirp_duration", fallback="1/symbol_rate")
        self._chirp_bandwidth_str = cfg.get(
            "chirp_bandwidth", fallback="symbol_rate")

        self.pulse_width = cfg.getfloat("pulse_width", fallback=1.0)

        cfg = config["Receiver"]
        self.equalizer = cfg.get("equalizer", fallback="NONE").upper()

        cfg = config['Frame']
        self.number_preamble_symbols = cfg.getint(
            "number_preamble_symbols", fallback=0)
        self.number_data_symbols = cfg.getint("number_data_symbols")
        self.number_postamble_symbols = cfg.getint(
            "number_postamble_symbols", fallback=0)
        self._pilot_symbol_rate_str = cfg.get(
            "pilot_symbol_interval", fallback="symbol_rate")

        self.guard_interval = cfg.getfloat("guard_interval", fallback=0)

        self.sampling_rate = self.symbol_rate * self.oversampling_factor
        self.bits_per_symbol = int(np.log2(self.modulation_order))
        self.bits_in_frame = self.number_data_symbols * self.bits_per_symbol

        self._check_params()

    def _check_params(self) -> None:
        """checks the validity of the parameters."""
        top_header = 'ERROR reading PSK/QAM modem parameters'

        #######################
        # check modulation parameters
        msg_header = top_header + ', Section "Modulation", '

        if (self.modulation_is_complex and
                self.modulation_order not in ParametersPskQam.modulation_order_complex_val):
            raise ValueError(
                f'{msg_header}complex modulation order ({self.modulation_order}) not supported')

        if (not self.modulation_is_complex and
                self.modulation_order not in ParametersPskQam.modulation_order_real_val):
            raise ValueError(
                f'{msg_header}real modulation order ({self.modulation_order}) not supported')

        if self.symbol_rate <= 0:
            raise ValueError(
                f'{msg_header}symbol_rate ({self.symbol_rate}) must be > 0')

        if self.oversampling_factor < 1:
            raise ValueError(
                f'{msg_header}oversampling_factor ({self.oversampling_factor}) must be >= 1')

        if self.filter_type not in ParametersPskQam.filter_type_val:
            raise ValueError(
                f'{msg_header}filter type (' +
                self.filter_type +
                ') not supported')

        if self.filter_type == "RAISED_COSINE" or self.filter_type == "ROOT_RAISED_COSINE":
            if self.roll_off_factor < 0 or self.roll_off_factor > 1:
                raise ValueError(
                    f'{msg_header}roll off factor ({self.roll_off_factor}) must be in interval [0, 1]')

            if self.filter_length_in_symbols < 2:
                raise ValueError(
                    f'{msg_header}filter length ({self.filter_length_in_symbols}) must be >1')

        if self.filter_type == "RECTANGULAR":
            if self.pulse_width <= 0 or self.pulse_width > 1:
                raise ValueError(
                    f'{msg_header}pulse width ({self.pulse_width}) must be in the interval (0, 1]')
            samples_in_rect = self.pulse_width * self.oversampling_factor
            if not float(int(samples_in_rect)) == samples_in_rect:
                raise ValueError(
                    f'{msg_header}pulse width * oversampling factor must be an integer')

            self.bandwidth = self.symbol_rate / self.pulse_width

        elif self.filter_type == "FMCW":
            self._chirp_duration_str = self._chirp_duration_str.replace(
                "symbol_rate", "self.symbol_rate")
            try:
                self.chirp_duration = eval(self._chirp_duration_str)
            except (SyntaxError, NameError):
                raise ValueError(
                    f'{msg_header}chirp duration ({self._chirp_duration_str}) cannot be evaluated')
            if self.chirp_duration <= 0:
                raise ValueError(
                    f'{msg_header}chirp duration ({self._chirp_duration_str}) must be >0')
            self.filter_length_in_symbols = self.chirp_duration * self.symbol_rate

            self._chirp_bandwidth_str = self._chirp_bandwidth_str.replace(
                "symbol_rate", "self.symbol_rate")
            try:
                self.chirp_bandwidth = eval(self._chirp_bandwidth_str)
            except (SyntaxError, NameError):
                raise ValueError(
                    f'{msg_header}chirp bandwidth ({self._chirp_bandwidth_str}) cannot be evaluated')

            if self.sampling_rate < self.chirp_bandwidth:
                self.oversampling_factor = math.ceil(
                    abs(self.chirp_bandwidth) / self.symbol_rate)
                self.sampling_rate = self.oversampling_factor * self.symbol_rate
                warnings.warn(f"{msg_header}chirp bandwidth ({self._chirp_bandwidth_str} Hz) must be <= sampling rate"
                              f"({self.sampling_rate} sps).\n"
                              f"Oversampling factor was changed to {self.oversampling_factor}")
            self.bandwidth = self.chirp_bandwidth

        else:
            self.bandwidth = self.symbol_rate

        #############################
        # check receiver parameters
        msg_header = top_header + ', Section "receiver", '
        if self.equalizer not in ParametersPskQam.equalizer_val:
            raise ValueError(f"{msg_header}equalizer ('{self.equalizer}') not supported")


        #############################
        # check frame parameters
        msg_header = top_header + ', Section "Frame", '

        if self.number_preamble_symbols < 0:
            raise ValueError(
                f'{msg_header}number of preamble symbols ({self.number_preamble_symbols}) must be >= 0')

        if self.number_data_symbols <= 0:
            raise ValueError(
                f'{msg_header}number of postamble symbols ({self.number_postamble_symbols}) must be > 0')

        if self.number_postamble_symbols < 0:
            raise ValueError(
                f'{msg_header}number of postamble symbols ({self.number_postamble_symbols}) must be >= 0')

        if self.guard_interval < 0:
            raise ValueError(
                msg_header +
                'guard interval ({:f}) must be >= 0'.format(
                    self.guard_interval))

        self._pilot_symbol_rate_str = self._pilot_symbol_rate_str.replace(
            "symbol_rate", "self.symbol_rate")
        self._pilot_symbol_rate_str = self._pilot_symbol_rate_str.replace(
            "chirp_bandwidth", "self.chirp_bandwidth")
        self._pilot_symbol_rate_str = self._pilot_symbol_rate_str.replace(
            "chirp_duration", "self.chirp_duration")
        try:
            self.pilot_symbol_rate = eval(self._pilot_symbol_rate_str)
        except (SyntaxError, NameError):
            raise ValueError(
                f'{msg_header}pilot symbol rate ({self._pilot_symbol_rate_str}) cannot be evaluated')
        if self.pilot_symbol_rate <= 0:
            raise ValueError(
                f'{msg_header}pilot symbol_rate ({self.pilot_symbol_rate}) must be > 0')
