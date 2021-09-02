import configparser
import numpy as np


class ParametersGeneral(object):
    """This class implements the parser for the parameters that control the simulation flow.

    Attributes:
        snr_type_val:
        metric_val: list of the allowed values for parameters 'snr_type' and 'metric'
        seed: seed for random number generation

        min_num_drops (int):
        max_num_drops (int):
        drop_length (float):
        metric (str):
        confidence_level (float):
        confidence_margin (float): Parameters that control the simulation flow.
            More details are given in '_settings/settings_general.ini'.

        snr_type (str):
        snr_vector (np.ndarray): Parameters that control the loop over different
            noise levels. More details are given in '_settings/settings_general.ini'.

        verbose (bool):
        plot
        calc_spectrum_tx (bool):
        calc_stft_tx (bool):
        spectrum_fft_size (int): Üarameters that control the statistics that will
            be generated. More details are given in '_settings/settings_general.ini'.
    """

    snr_type_val = ['EB/N0(DB)', 'ES/N0(DB)', 'CUSTOM']
    metric_val = ['BER', 'FER']

    def __init__(self) -> None:
        # Random Number parameters
        self.seed = 0

        # Drops parameters
        self.min_num_drops = 0
        self.max_num_drops = 0
        self.drop_length = 0.

        self.confidence_level = 0.
        self.confidence_margin = 0.
        self.confidence_metric = ""

        # Noise loop parameters
        self.snr_type = ""
        self.snr_vector = np.array([])

        # statistics parameters
        self.verbose = False
        self.plot = False
        self.calc_spectrum_tx = False
        self.calc_stft_tx = False
        self.calc_spectrum_rx = False
        self.calc_stft_rx = False
        self.spectrum_fft_size = 0
        self.calc_theory = 0.

    def read_params(self, file_name: str) -> None:
        """This method reads and checks the validity of all the parameters from the file 'file_name'."""
        config = configparser.ConfigParser()
        config.read(file_name)

        cfg = config['RandomNumber']
        self.seed = cfg.getint("seed")

        cfg = config['Drops']
        self.min_num_drops = cfg.getint("min_num_drops")
        self.max_num_drops = cfg.getint("max_num_drops")
        self.drop_length = cfg.getfloat("duration")
        self.confidence_metric = cfg.get("confidence_metric").upper()
        self.confidence_level = cfg.getfloat("confidence_level")
        self.confidence_margin = cfg.getfloat("confidence_margin")

        cfg = config['NoiseLoop']
        self.snr_type = cfg.get("snr_type").upper()
        self._snr_vector_str = cfg.get("snr_vector")

        cfg = config['Statistics']
        self.verbose = cfg.getboolean("verbose")
        self.plot = cfg.getboolean("plot")
        self.calc_spectrum_tx = cfg.getboolean("calc_spectrum_tx")
        self.calc_stft_tx = cfg.getboolean("calc_stft_tx")
        self.calc_spectrum_rx = cfg.getboolean("calc_spectrum_rx")
        self.calc_stft_rx = cfg.getboolean("calc_stft_rx")
        self.spectrum_fft_size = cfg.getint("spectrum_fft_size")
        self.calc_theory = cfg.getboolean("calc_theory")

        self._check_params()

    def _check_params(self) -> None:
        """ This method validates all the general-purpose simulation parameters
        """

        top_header = 'ERROR reading General parameters'

        #######################
        # check drop parameters
        msg_header = top_header + ', Section "Drop", '

        if self.min_num_drops < 1:
            raise ValueError(
                msg_header +
                'min_num_drops ({:d}) must be larger than 0'.format(
                    self.min_num_drops))

        if self.max_num_drops < 1:
            raise ValueError(
                msg_header +
                'min_num_drops ({:d}) must be larger than 0'.format(
                    self.min_num_drops))

        if self.min_num_drops > self.max_num_drops:
            raise ValueError(msg_header +
                             'min_num_drops ({:d}) cannot be larger than max_num_drops({:d})'.format(
                                 self.min_num_drops, self.max_num_drops))

        if self.drop_length < 0:
            raise ValueError(
                msg_header +
                'duration ({:f}) cannot be less than 0'.format(
                    self.drop_length))

        if self.confidence_metric not in ParametersGeneral.metric_val:
            raise ValueError(
                msg_header +
                'metric (' +
                self.confidence_metric +
                ' is not a valid option')

        if self.confidence_level <= 0 or self.confidence_level >= 1:
            raise ValueError(msg_header +
                             'confidence_level (' +
                             str(self.confidence_level) +
                             ') must be between 0 and 1')

        if self.confidence_margin < 0:
            raise ValueError(msg_header +
                             'confidence_margin (' +
                             str(self.confidence_margin) +
                             ') must be >= 0')

        #############################
        # check noise loop parameters
        msg_header = top_header + ', Section "NoiseLoop", '

        if self.snr_type not in ParametersGeneral.snr_type_val:
            raise ValueError(
                msg_header +
                'snr_type (' +
                self.snr_type +
                ' is not a valid option')

        try:
            self.snr_vector = np.array(eval(self._snr_vector_str))
        except Exception:
            raise ValueError(
                msg_header +
                'snr_vector (' +
                self.snr_type +
                ') could not be interpreted')

        #############################
        # check statistics parameters
        msg_header = top_header + ', Section "Statistics", '

        if self.spectrum_fft_size <= 0:
            raise ValueError(msg_header +
                             'periodogram_fft_size (' +
                             str(self.spectrum_fft_size) +
                             ' must be > 0')
