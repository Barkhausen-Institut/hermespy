import numpy as np

import configparser
from parameters_parser.parameters_waveform_generator import ParametersWaveformGenerator


class ParametersChirpFsk(ParametersWaveformGenerator):
    """This class implements the parser of a chirp FSK modem

    Attributes:
        chirp_duration:
        chirp_bandwidth:
        modulation_order:
        freq_difference:
        oversampling_factor: Parameters that control the modulated waveform.
            More details are given in '_settings/settings_chirp_fsk.ini'.
        number_pilot_chirps:
        number_data_chirps:
        guard_interval: Parameters that describe the transmission frame.
            More details are given in '_settings/settings_chirp_fsk.ini'.
        bits_in_frame:
        bits_per_symbol:
    """

    def __init__(self) -> None:
        """Creates a parsing object, that will manage the transceiver parameters."""

        super().__init__()

        # Modulation parameters
        self.modulation_order = 0
        self.chirp_duration = 0.
        self.chirp_bandwidth = 0.
        self.freq_difference = 0.

        self.oversampling_factor = 0

        # Frame parameters
        self.number_pilot_chirps = 0
        self.number_data_chirps = 0
        self.guard_interval = 0.
        self.bits_per_symbol = 0
        self.bits_in_frame = 0

    def read_params(self, file_name: str) -> None:
        """Reads the modem parameters contained in the configuration file 'file_name'."""
        super().read_params(file_name)

        config = configparser.ConfigParser()
        config.read(file_name)

        cfg = config['Modulation']
        self.modulation_order = cfg.getint("modulation_order")
        self.chirp_duration = cfg.getfloat("chirp_duration")
        self.chirp_bandwidth = cfg.getfloat("chirp_bandwidth")
        self.freq_difference = cfg.getfloat("freq_difference")

        self.oversampling_factor = cfg.getint("oversampling_factor")
        self.sampling_rate = self.chirp_bandwidth * self.oversampling_factor

        cfg = config['Frame']
        self.number_pilot_chirps = cfg.getint("number_pilot_chirps")
        self.number_data_chirps = cfg.getint("number_data_chirps")
        self.guard_interval = cfg.getfloat("guard_interval")
        self.bits_per_symbol = int(np.log2(self.modulation_order))
        self.bits_in_frame = self.number_data_chirps * self.bits_per_symbol

        self._check_params()

    def _check_params(self) -> None:
        """checks the validity of the parameters"""
        top_header = 'ERROR reading chirp FSK modem parameters'

        #######################
        # check modulation parameters
        msg_header = top_header + ', Section "Modulation", '

        if self.modulation_order <= 0 or (
                self.modulation_order & (self.modulation_order - 1)) != 0:
            raise ValueError(
                msg_header +
                'modulation_order must be a positive power of two')

        if self.chirp_duration <= 0:
            raise ValueError(
                msg_header +
                'chirp_duration ({:f}) must be > 0'.format(
                    self.chirp_duration))

        if self.chirp_bandwidth <= 0:
            raise ValueError(
                msg_header +
                'chirp_bandwidth ({:f}) must be > 0'.format(
                    self.chirp_bandwidth))

        if self.freq_difference < 0 or self.freq_difference >= self.chirp_bandwidth:
            raise ValueError(msg_header +
                             'freq_difference ({:f}) must be less than chirp_bandwidth'.format(self.freq_difference))

        if self.modulation_order * self.freq_difference > self.chirp_bandwidth:
            raise ValueError(
                msg_header +
                'bandwidth of modulated signal is larger than chirp bandwidth')

        if self.oversampling_factor < 1:
            raise ValueError(
                msg_header +
                'oversampling_factor ({:d}) must be >= 1'.format(
                    self.oversampling_factor))

        #############################
        # check frame parameters
        msg_header = top_header + ', Section "Frame", '

        if self.number_pilot_chirps < 0:
            raise ValueError(
                msg_header +
                'number_pilot_chirps ({:d}) must be >= 0'.format(
                    self.number_pilot_chirps))

        if self.number_data_chirps < 0:
            raise ValueError(
                msg_header +
                'number_data_chirps ({:d}) must be >= 0'.format(
                    self.number_data_chirps))

        if self.guard_interval < 0:
            raise ValueError(
                msg_header +
                'guard interval ({:f}) must be >= 0'.format(
                    self.guard_interval))
