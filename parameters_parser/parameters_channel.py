import configparser

import numpy as np

from parameters_parser.parameters_rx_modem import ParametersRxModem
from parameters_parser.parameters_tx_modem import ParametersTxModem

class ParametersChannel:
    """This class implements the parser of the channel model parameters.

    Attributes:
        multipath_model_val list(str): list of valid channel multipath models.

        device_type (str): defines which device type the modem ist
        cov_matrix (np.ndarray): Correlation matrix for MIMO calculation
        antenna_spacing (float): Ratio between antenna distance and lambda/2

    """

    multipath_model_val = [
        "NONE",
        "STOCHASTIC",
        "COST259",
        "EXPONENTIAL",
        "QUADRIGA",
        "5G_TDL"]
    cost_259_type_val = ["TYPICAL_URBAN", "RURAL_AREA", "HILLY_TERRAIN"]
    exponential_truncation = 1e-5  # truncate exponential profile for values less than this
    supported_tdl = ["A", "B", "C", "D", "E"]
    correlation_type_val = ["LOW", "MEDIUM", "MEDIUM_A", "HIGH", "CUSTOM"]
    alpha_val = {"LOW": 0, "MEDIUM": 0.3, "MEDIUM_A": 0.3, "HIGH": 0.9}
    beta_val = {"LOW": 0, "MEDIUM": 0.9, "MEDIUM_A": 0.3874, "HIGH": 0.9}

    def __init__(self, rx_modem_params: ParametersRxModem, tx_modem_params: ParametersTxModem) -> None:
        """Creates a parsing object, that will manage the channel parameters."""
        self.multipath_model = "NONE"
        self.delays = np.array([])
        self.power_delay_profile_db = np.array([])
        self.power_delay_profile = np.array([])
        self.k_factor_rice = np.array([])
        self.velocity = np.array([])
        self.attenuation_db = 0.
        self.gain = 0.
        self.los_doppler_factor = 1.
        self._tx_correlation = "LOW"
        self._rx_correlation = "LOW"
        self._tx_custom_correlation = 0.
        self._rx_custom_correlation = 0.
        self.rms_delay_spread = 0.
        self.tx_cov_matrix = np.array([])
        self.rx_cov_matrix = np.array([])

        self.params_rx_modem = rx_modem_params
        self.params_tx_modem = tx_modem_params
        # parameters for COST 259 channel model only
        # previous parameters are derived from them
        self.cost_259_type = ""
        # parameter relevant for COST 259 hilly-terrain model only (angle of
        # LOS component)
        self.los_theta_0 = 0.

        # parameters for 5G Phy model, TDL only
        self.tdl_type = ""
        self.delays_normalized = np.array([])

        # parameters for exponential channel model only
        # previous parameters are derived from them
        self.tap_interval = 0.
        self.rms_delay = 0.

        # parameters for Quadriga
        self.matlab_or_octave = 0.
        self.scenario_qdg = 0.
        self.antenna_kind = 0.
        self.path_quadriga_src = 0.

    def read_params(self, section: configparser.SectionProxy):
        """Reads channel parameters of a given config file.

        Args:
            section (configparser.SectionProxy): Section in the file to read the
                parameters from.
        """
        self._rx_correlation = section.get("rx_correlation", fallback="LOW").upper()
        self._tx_correlation = section.get("tx_correlation", fallback="LOW").upper()

        self._rx_custom_correlation = section.getfloat("rx_custom_correlation", fallback=0.)
        self._tx_custom_correlation = section.getfloat("tx_custom_correlation", fallback=0.)

        self.multipath_model = section.get(
            "multipath_model", fallback='none').upper()

        if self.multipath_model == "STOCHASTIC":
            self.delays = section.get("delays", fallback='0')
            self.delays = np.fromstring(self.delays, sep=',')

            self.power_delay_profile_db = section.get(
                "power_delay_profile_db", fallback='0')
            self.power_delay_profile_db = np.fromstring(
                self.power_delay_profile_db, sep=',')

            k_factor_rice_db_str = section.get("k_rice_db", fallback='-inf')
            k_factor_rice_db = np.fromstring(k_factor_rice_db_str, sep=',')
            self.k_factor_rice = 10 ** (k_factor_rice_db / 10)
            self.los_doppler_factor = section.getfloat("los_doppler_factor", fallback=1.)

        if self.multipath_model == "COST259":
            self.cost_259_type = section.get(
                "cost_type", fallback='typical_urban').upper()

        if self.multipath_model == "5G_TDL":
            self.tdl_type = section.get(
                "tdl_type", fallback="A").upper()
            self.rms_delay = section.getfloat("rms_delay")

        if self.multipath_model == "EXPONENTIAL":
            self.tap_interval = section.getfloat("tap_interval")
            self.rms_delay = section.getfloat("rms_delay")

        self.attenuation_db = section.getfloat("attenuation_db", fallback=0)

    def check_params(self):
        """Checks the validity of the read parameters."""
        if self._rx_correlation not in self.correlation_type_val:
            raise ValueError(f'Correlation {self._rx_correlation} not supported')
        if self._tx_correlation not in self.correlation_type_val:
            raise ValueError(f'Correlation {self._tx_correlation} not supported')

        if not (0 <= self._rx_custom_correlation <= 1):
            raise ValueError('custom correlation must be between 0 and 1')
        if not (0 <= self._tx_custom_correlation <= 1):
            raise ValueError('custom correlation must be between 0 and 1')

        if self._rx_correlation != "CUSTOM" or self._tx_correlation != "CUSTOM":
            if (self.params_tx_modem.number_of_antennas > 4 or
               self.params_rx_modem.number_of_antennas > 4):
                raise ValueError('number of antennas must be maximum 4.')

            no_tx_antennas = self.params_tx_modem.number_of_antennas
            no_rx_antennas = self.params_rx_modem.number_of_antennas

            if ((no_tx_antennas & (no_tx_antennas - 1)) == 1 or
               (no_rx_antennas & (no_rx_antennas - 1)) == 1):
                raise ValueError("number of antennas must be power of 2")

        self._calculate_correlation_matrices()

        if self.multipath_model not in ParametersChannel.multipath_model_val:
            raise ValueError(
                "multipath_model '" +
                self.multipath_model +
                "' not supported")

        if self.multipath_model == "STOCHASTIC":
            if self.delays.shape != self.power_delay_profile_db.shape:
                raise ValueError(
                    "'power_delay_profile' must have the same length as 'delays'")

            if self.delays.shape != self.k_factor_rice.shape:
                raise ValueError(
                    "'power_delay_profile' must have the same length as 'delays'")

        elif self.multipath_model == "COST259":
            if self.cost_259_type not in ParametersChannel.cost_259_type_val:
                raise ValueError(
                    'COST 259 type (' + self.cost_259_type + ') not supported')
            elif self.cost_259_type == 'TYPICAL_URBAN':
                self.delays = np.asarray([0, .217, .512, .514, .517, .674, .882, 1.230, 1.287, 1.311, 1.349, 1.533,
                                          1.535, 1.622, 1.818, 1.836, 1.884, 1.943, 2.048, 2.140]) * 1e-6
                self.power_delay_profile_db = np.asarray([-5.7, - 7.6, -10.1, -10.2, -10.2, -11.5, -13.4, -16.3, -16.9,
                                                          -17.1, -17.4, -19.0, -19.0, -19.8, -21.5, -21.6, -22.1, -22.6,
                                                          -23.5, -24.3])
                self.k_factor_rice = np.zeros(self.delays.shape)

            elif self.cost_259_type == 'RURAL_AREA':
                self.delays = np.asarray(
                    [0, .042, .101, .129, .149, .245, .312, .410, .469, .528]) * 1e-6
                self.power_delay_profile_db = np.asarray([-5.2, -6.4, -8.4, -9.3, -10.0, -13.1, -15.3, -18.5, -20.4,
                                                          -22.4])
                self.k_factor_rice = np.zeros(self.delays.shape)

            elif self.cost_259_type == 'HILLY_TERRAIN':
                self.delays = np.asarray([0, .356, .441, .528, .546, .609, .625, .842, .916, .941, 15.0, 16.172, 16.492,
                                          16.876, 16.882, 16.978, 17.615, 17.827, 17.849, 18.016]) * 1e-6
                self.power_delay_profile_db = np.asarray([-3.6, -8.9, -10.2, -11.5, -11.8, -12.7, -13.0, -16.2, -17.3,
                                                          -17.7, -17.6, -22.7, -24.1, -25.8, -25.8, -26.2, -29.0, -29.9,
                                                          -30.0, -30.7])
                self.k_factor_rice = np.hstack(
                    (np.inf, np.zeros(self.delays.size - 1)))
                self.los_theta_0 = np.arccos(.7)

            self.multipath_model = "STOCHASTIC"

        # cf. https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/14.00.00_60/tr_138901v140000p.pdf
        elif self.multipath_model == "5G_TDL":
            if self.tdl_type not in self.supported_tdl:
                raise ValueError(
                    'TDL type (' + self.tdl_type + ') not supported')
            if self.tdl_type == "A":
                self.delays_normalized = np.array([0, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750, 0.7618,
                                        1.5375, 1.8978, 2.2242, 2.1717, 2.4942, 2.5119, 3.0582,
                                        4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586])
                self.power_delay_profile_db = np.array([-13.4, 0, -2.2, -4, -6, -8.2, -9.9, -10.5,
                                                        -7.5, -15.9, -6.6, -16.7, -12.4, -15.2, -10.8,
                                                        -11.3, -12.7, -16.2, -18.3, -18.9, -16.6, -19.9, -29.7])
                self.k_factor_rice = np.zeros(self.delays_normalized.shape)

            elif self.tdl_type == "B":
                self.delays_normalized = np.array([0, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986,
                                        0.3752, 0.5055, 0.3681, 0.3697, 0.5700,
                                        0.5283, 1.1021, 1.2756, 1.5474, 1.7842,
                                        2.0169, 2.8294, 3.0219, 3.6187, 4.1067,
                                        4.2790, 4.7834])
                self.power_delay_profile_db = np.array([0, -2.2, -4, -3.2, -9.8,
                                                        -3.2, -3.4, -5.2,
                                                        -7.6, -3, -8.9, -9,
                                                        -4.8, -5.7, -7.5, -1.9,
                                                        -7.6, -12.2, -9.8, -11.4,
                                                        -14.9, -9.2, -11.3])
                self.k_factor_rice = np.zeros(self.delays_normalized.shape)

            elif self.tdl_type == "C":
                self.delays_normalized = np.array([0, 0.2099, 0.2219, 0.2329, 0.2176,
                                        0.6366, 0.6448, 0.6560, 0.6584, 0.7935,
                                        0.8213, 0.9336, 1.2285, 1.3083, 2.1704,
                                        2.7105, 4.2589, 4.6003, 5.4902, 5.6077,
                                        6.3065, 6.6374, 7.0427, 8.6523])
                self.power_delay_profile_db = np.array([-4.4, -1.2, -3.5, -5.2, -2.5,
                                                        0, -2.2, -3.9, -7.4, -7.1, -10.7,
                                                        -11.1, -5.1, -6.8, -8.7, -13.2,
                                                        -13.9, -13.9, -15.8, -17.1, -16,
                                                        -15.7, -21.6, -22.8])
                self.k_factor_rice = np.zeros(self.delays_normalized.shape)

            elif self.tdl_type == "D":
                self.delays_normalized = np.array([0, 0.035, 0.612, 1.363, 1.405,
                                        1.804, 2.596, 1.775, 4.042,
                                        7.937, 9.424, 9.708, 12.525])
                self.power_delay_profile_db = np.array([-13.5, -18.8, -21,
                                                        -22.8, -17.9, -20.1,
                                                        -21.9, -22.9, -27.8,
                                                        -23.6, -24.8, -30.0,
                                                        -27.7])
                self.k_factor_rice = np.zeros(self.delays_normalized.shape)
                self.k_factor_rice[0] = 13.3
                self.los_doppler_factor = 0.7  # defined in the standard

            elif self.tdl_type == "E":
                self.delays_normalized = np.array([0, 0.5133, 0.5440, 0.5630, 0.5440,
                                        0.7112, 1.9092, 1.9293, 1.9589,
                                        2.6426, 3.7136, 5.4524, 12.0034,
                                        20.6519])
                self.power_delay_profile_db = np.array([-22.03, -15.8, -18.1,
                                                        -19.8, -22.9, -22.4,
                                                        -18.6, -20.8, -22.6,
                                                        -22.3, -25.6, -20.2,
                                                        -29.8, -29.2])
                self.k_factor_rice = np.zeros(self.delays_normalized.shape)
                self.k_factor_rice[0] = 22
                self.los_doppler_factor = 0.7  # defined in the standard

            # scale delays as they are normalized
            self.delays = self.rms_delay * self.delays_normalized

        elif self.multipath_model == "EXPONENTIAL":
            # calculate the decay exponent alpha based on an infinite power delay profile, in which case
            # rms_delay = exp(-alpha/2)/(1-exp(-alpha)), cf. geometric distribution
            # Truncate the distributions for paths whose average power is very
            # small (less than exponential_truncation)
            rms_norm = self.rms_delay / self.tap_interval
            alpha = -2 * \
                np.log((-1 + np.sqrt(1 + 4 * rms_norm ** 2)) / (2 * rms_norm))
            max_delay_in_samples = - \
                int(np.ceil(np.log(ParametersChannel.exponential_truncation) / alpha))
            self.delays = np.arange(
                max_delay_in_samples + 1) * self.tap_interval
            self.power_delay_profile_db = 10 * \
                np.log10(np.exp(-alpha * np.arange(max_delay_in_samples + 1)))
            self.k_factor_rice = np.zeros(self.delays.shape)

            self.multipath_model = "STOCHASTIC"

        ####
        elif self.multipath_model == "QUADRIGA":
            # see quadriga_doc
            self.multipath_model = "QUADRIGA"
        ###

        if self.multipath_model != "NONE" and self.multipath_model != "QUADRIGA":
            self.power_delay_profile = 10 ** (self.power_delay_profile_db / 10)
            self.power_delay_profile = self.power_delay_profile / \
                sum(self.power_delay_profile)

        self.gain = 10 ** (-self.attenuation_db / 20)

    def _calculate_correlation_matrices(self):
        self.tx_cov_matrix = self._calculate_one_matrix(
            self.params_tx_modem.device_type,
            self.params_tx_modem.number_of_antennas,
            self.params_tx_modem.antenna_spacing, "tx")

        self.rx_cov_matrix = self._calculate_one_matrix(
            self.params_rx_modem.device_type,
            self.params_rx_modem.number_of_antennas,
            self.params_rx_modem.antenna_spacing, "rx")

    def _calculate_one_matrix(self,
                              device_type: str,
                              no_antennas: int, antenna_spacing: float,
                              modem_type: str) -> np.ndarray:

        correlation = ""
        if modem_type == "tx":
            correlation = self._tx_correlation
        else:
            correlation = self._rx_correlation

        if correlation == "CUSTOM":
            if modem_type == "tx":
                a = self._tx_custom_correlation
            else:
                a = self._rx_custom_correlation
        elif device_type == "BASE_STATION":
            a = self.alpha_val[correlation]
        elif device_type == "UE":
            a = self.beta_val[correlation]

        cov_matrix = np.array([])

        if correlation == "CUSTOM":
            cov_matrix = np.eye(no_antennas, no_antennas)
            for i in range(no_antennas):
                for j in range(no_antennas):
                    cov_matrix[i, j] = a ** ((abs(i-j)) * antenna_spacing)
        else:
            if no_antennas == 1:
                cov_matrix = np.ones((1, 1))
            elif no_antennas == 2:
                cov_matrix = np.array([[1, a],
                                       [a, 1]])
            elif no_antennas >= 4:
                cov_matrix = np.array([[1, a ** (1 / 9), a ** (4 / 9), a],
                                       [a**(1 / 9), 1, a**(1 / 9), a**(4 / 9)],
                                       [a**(4 / 9), a**(1 / 9), 1, a**(1 / 9)],
                                       [a, a**(4 / 9), a**(1 / 9), 1]])
        return cov_matrix
