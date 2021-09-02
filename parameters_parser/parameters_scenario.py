import configparser
import os
from typing import List, Any

from parameters_parser.parameters_tx_modem import ParametersTxModem
from parameters_parser.parameters_rx_modem import ParametersRxModem
from parameters_parser.parameters_channel import ParametersChannel
from parameters_parser.parameters_quadriga import ParametersQuadriga


class ParametersScenario:
    """This class implements the parser for the parameters that describe the simulation scenario.

    Attributes:
        number_of_tx_modems (int):
        number_of_rx_modems (int): number of transmit and receive modems in the scenario

        tx_modem_params (list(ParametersModem)): list of length 'number_of_tx_modems'
            containing the parameter parsers of all transmit modems.

        rx_modem_params (list(ParametersModem)): list of length 'number_of_rx_modems'
            containing the parameter parsers of all receive modems.

        channel_model_params (list[list(ParametersModem)]): 2D- list of size
            'number_of_rx_modems' x 'number_of_tx_modems' containing the parameter parsers
            of the channel models. channel_model_params[i][j] contains the channel
            model from transmit modem 'j' to receive modem 'i'.
    """

    def __init__(self) -> None:
        self.number_of_tx_modems = 0
        self.number_of_rx_modems = 0
        self.tx_modem_params: List[ParametersTxModem] = []
        self.rx_modem_params: List[ParametersRxModem] = []

        self.channel_model_params: List[List[Any]] = []

    def read_params(self, file_name: str) -> None:
        quadriga_channel = False

        config = configparser.ConfigParser()
        config.read(file_name)

        sections = config.sections()

        # read TxModem parameters
        while 'TxModem_' + str(self.number_of_tx_modems + 1) in sections:
            cfg = config['TxModem_' + str(self.number_of_tx_modems + 1)]

            tx_modem_params = ParametersTxModem()
            tx_modem_params.read_params(cfg)
            tx_modem_params.id = self.number_of_tx_modems + 1
            self.tx_modem_params.append(tx_modem_params)

            self.number_of_tx_modems += 1

        # read RxModem parameters
        while 'RxModem_' + str(self.number_of_rx_modems + 1) in sections:
            cfg = config['RxModem_' + str(self.number_of_rx_modems + 1)]

            rx_modem_params = ParametersRxModem()
            rx_modem_params.read_params(cfg)
            rx_modem_params.id = self.number_of_rx_modems + 1
            self.rx_modem_params.append(rx_modem_params)

            self.number_of_rx_modems += 1

        # read Channel parameters
        channel_params: Any
        for rx_modem_idx in range(1, self.number_of_rx_modems + 1):

            ch_model_rx: List[Any] = []
            for tx_modem_idx in range(1, self.number_of_tx_modems + 1):
                section_name = 'Channel_' + \
                    str(tx_modem_idx) + '_to_' + str(rx_modem_idx)
                if section_name not in sections:
                    raise ValueError("ERROR reading Scenario parameters, no instance of Channel" +
                                     "between TX {:d} and RX {:d} found".format(tx_modem_idx, rx_modem_idx))

                cfg = config[section_name]
                if cfg.get("multipath_model",
                           fallback='none').upper() == "QUADRIGA":
                    channel_params = ParametersQuadriga(file_name)
                    quadriga_channel = True
                elif quadriga_channel is True:
                    raise ValueError(
                        "If quadriga channel is set, all channels need to be of type quadriga.")
                else:
                    channel_params = ParametersChannel(
                        self.rx_modem_params[rx_modem_idx - 1],
                        self.tx_modem_params[tx_modem_idx - 1]
                    )

                channel_params.read_params(cfg)
                ch_model_rx.append(channel_params)

            self.channel_model_params.append(ch_model_rx)

        self._check_params(os.path.dirname(file_name))

    def _check_params(self, param_path: str) -> None:
        """ This method validates all the scenario parameters
        """
        top_header = 'ERROR reading Scenario parameters, '

        if self.number_of_tx_modems < 1:
            raise ValueError(top_header + "no instance of TxModem found")

        if self.number_of_rx_modems < 1:
            raise ValueError(
                "ERROR reading Scenario parameters, no instance of RxModem found")

        #######################
        # check TxModem parameters
        for idx in range(self.number_of_tx_modems):
            try:
                self.tx_modem_params[idx].check_params(param_path)
            except ValueError as detail:
                msg_header = top_header + \
                    'Section "TxModem ' + str(idx) + '", '
                raise ValueError(msg_header + detail.args[0])

        #######################
        # check RxModem parameters
        for idx in range(self.number_of_rx_modems):
            try:
                self.rx_modem_params[idx].check_params(self.tx_modem_params, param_path)
            except ValueError as detail:
                msg_header = top_header + \
                    ', Section "RxModem ' + str(idx) + '", '
                raise ValueError(msg_header + detail.args[0])

        #######################
        # check Channel parameters
        for idx_tx in range(self.number_of_tx_modems):
            for idx_rx in range(self.number_of_rx_modems):

                channel_params = self.channel_model_params[idx_rx][idx_tx]
                try:
                    channel_params.check_params()
                except ValueError as detail_msg:
                    msg_header = top_header + ' Section "Channel_ ' + \
                        str(idx_tx + 1) + '_to_' + str(idx_rx + 1) + '",'
                    raise ValueError(msg_header + detail_msg.args[0])
