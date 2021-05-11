import os
from typing import List, Tuple
from shutil import which

import numpy as np

from modem.modem import Modem
from parameters_parser.parameters_quadriga import ParametersQuadriga


MATLAB_INSTALLED = False
OCTAVE_INSTALLED = False

try:
    import matlab.engine
    MATLAB_INSTALLED = True
except ImportError:
    print("""Warning! Module matlab.engine is not installed. If you want to
                use matlab for running Quadriga simulation, please install this
                package.""")
    pass

try:
    from oct2py import octave
    OCTAVE_INSTALLED = True
except ImportError:
    print("""Warning. oct2py is not installed. If you want to use
                octave for running Quadriga simulation, please install this package. """)
    pass

MATLAB_INSTALLED = MATLAB_INSTALLED and (which("matlab") is not None)
OCTAVE_INSTALLED = OCTAVE_INSTALLED and (which("octave") is not None)


class QuadrigaInterface:
    """Implements the direct interface between hermes QuadrigChannel/quadriga backend.

    It is important to mention, that in the hermes implementation channels are
    independent of each other, i.e. are associated with each transmitter/receiver
    modem pair. However, this is not the case for quadriga which creates one
    channel for all transmitter/receiver modem pairs. Therefore, we need to do
    a mapping between the QuadrigaChannel objects for all transmitter/receiver
    modem pairs and the Quadriga simulation which runs in the background.

    This mapping is done in that class.

    Attributes:
        api_script_path(str): Path of the quadriga script to start.
        tx_modem_ids(List[int]): List of transmitter modems saved as ids.
        rx_modem_ids(List[int]): List of receiver modems saved as ids.
    """

    def __init__(self, params: ParametersQuadriga) -> None:
        self._quadriga_executor = params.quadriga_executor.lower()
        self.api_script_path = os.path.abspath(os.getcwd())
        self._path_quadriga_src = params.path_quadriga_src

        # We need this to track when a new drop actually starts.
        # As multiple QuadrigaChannels share one quadriga interface object,
        # we cannot say when a new drop starts and therefore, we cannot say
        # when quadroiga needs to be called again.
        self._new_drop = False

        if self._path_quadriga_src == "" and self._quadriga_executor == "matlab":
            raise ValueError("You must pass the quadriga src path!")
        elif self._quadriga_executor not in ["matlab", "octave"]:
            raise ValueError(
                "Quadriga executor must be either matlab or octave")

        if self._quadriga_executor == "matlab":
            if MATLAB_INSTALLED:
                self.engine = matlab.engine.start_matlab()
            else:
                raise Exception("""You want to execute the quadriga simulation with
                                   MATLAB, but the matlab.engine is not installed.""")
        else:
            if OCTAVE_INSTALLED:
                octave.addpath(self._path_quadriga_src)
            else:
                raise Exception("""You want to execute the quadriga simulation with
                                   OCTAE, but octave is not installed.""")

        # intialize empty parameters
        self.tx_modem_ids: List[int] = []
        self.rx_modem_ids: List[int] = []

        self._sampling_rate: List[float] = []
        self._number_tx: int = 0
        self._number_rx: int = 0
        self._carrier_frequency: List[float] = []
        self._tx_position: List[np.array] = []
        self._rx_position: List[np.array] = []
        self._tracks_length: List[float] = []
        self._tracks_speed: List[float] = []
        self._tracks_angle: List[float] = []
        self._tx_number_antennas: List[int] = []
        self._rx_number_antennas: List[int] = []

        self._parameters = params

    def update_quadriga_parameters(
            self, modem_tx: Modem, modem_rx: Modem) -> None:
        """Updates the quadriga parameters.

        Args:
            modem_tx(Modem): transmitting modem the QuadrigaChannel is created for.
            modem_rx(Modem): receiving modem the QuadrigaChannel is created for.
        """
        update_txmodem_list = False
        update_rxmodem_list = False

        update_txmodem_list = modem_tx.param.id not in self.tx_modem_ids
        update_rxmodem_list = modem_rx.param.id not in self.rx_modem_ids

        if update_txmodem_list:
            self.tx_modem_ids.append(modem_tx.param.id)
        if update_rxmodem_list:
            self.rx_modem_ids.append(modem_rx.param.id)

        self._update_modem_list(
            modem_tx, modem_rx, update_txmodem_list, update_rxmodem_list)

        update_txmodem_list = False
        update_rxmodem_list = False

    def init_drop(self, seed: float) -> None:
        """Initialization that needs to be performed for each drop.

        Args:
            seed (float): The seed to pass to quadriga.
        """
        self._seed = seed
        self._new_drop = True

    def get_impulse_response(
            self, modem_tx: Modem, modem_rx: Modem) -> Tuple[np.array, np.array]:
        """Get the impulse response betwen two modems.

        Args:
            modem_tx(Modem): Transmitting modem.
            modem_rx(Modem): Receiving modem.

        Returns:
            Tuple[np.array, np.array]: CIR and delay. Currently, only SISO.
        """
        self._launch_quadriga()
        if self._number_rx == 1 and self._number_tx == 1:
            cir_rx = self.cirs.path_impulse_responses
            tau_rx = self.cirs.tau
        else:
            cir_rx = self.cirs[modem_rx.param.id - 1,
                               modem_tx.param.id - 1].path_impulse_responses
            tau_rx = self.cirs[modem_rx.param.id -
                               1, modem_tx.param.id - 1].tau

        # make dimensions fit
        # oct2py automatically discards dimensions, i.e. (1,1,2,1) = (1,1,2,)
        if np.isscalar(cir_rx):
            cir_rx = np.array([[[[cir_rx]]]])
            tau_rx = np.array([[[[tau_rx]]]])

        for dim in range(4 - cir_rx.ndim):
            cir_rx = np.expand_dims(cir_rx, axis=cir_rx.ndim)
            tau_rx = np.expand_dims(tau_rx, axis=tau_rx.ndim)

        return cir_rx, tau_rx

    def _launch_quadriga(self) -> None:
        """Launches quadriga.

        We want to prevent multiple executions of quadriga.
        """
        if self._new_drop:
            self._new_drop = False

            self._set_executor_variables()

            if self._quadriga_executor == "matlab":
                self.engine.launch_quadriga_script(nargout=0)

                self.cirs = self.engine.workspace["cirs"]

            elif self._quadriga_executor == "octave":
                octave.eval("launch_quadriga_script")

                self.cirs = octave.pull("cirs")

    def _set_executor_variables(self) -> None:
        """Passes parameters from hermes to quadriga."""
        if self._quadriga_executor == "matlab":
            self.engine.workspace["sampling_rate"] = self._sampling_rate
            self.engine.workspace["carriers"] = matlab.double(
                self._carrier_frequency)
            self.engine.workspace["tx_position"] = matlab.double(
                self._tx_position)
            self.engine.workspace["rx_position"] = matlab.double(
                self._rx_position)

            self.engine.workspace["tx_antenna_kind"] = self._parameters.antenna_kind
            self.engine.workspace["rx_antenna_kind"] = self._parameters.antenna_kind
            self.engine.workspace["scenario_label"] = self._parameters.scenario_label
            self.engine.workspace["path_quadriga_src"] = self._path_quadriga_src
            self.engine.workspace["txs_number_antenna"] = matlab.double(
                self._tx_number_antennas)
            self.engine.workspace["rxs_number_antenna"] = matlab.double(
                self._rx_number_antennas)

            self.engine.workspace["number_tx"] = matlab.double(self._number_tx)
            self.engine.workspace["number_rx"] = matlab.double(self._number_rx)
            self.engine.workspace["tracks_speed"] = matlab.double(
                self._tracks_speed)
            self.engine.workspace["tracks_length"] = matlab.double(
                self._tracks_length)
            self.engine.workspace["tracks_angle"] = matlab.double(
                self._tracks_angle)
            self.engine.workspace["seed"] = matlab.double(self._seed)
        else:
            octave.push("sampling_rate", self._sampling_rate)
            octave.push("carriers", self._carrier_frequency)
            octave.push("tx_position", self._tx_position)
            octave.push("rx_position", self._rx_position)

            octave.push("scenario_label", self._parameters.scenario_label)
            octave.push("tx_antenna_kind", self._parameters.antenna_kind)
            octave.push("path_quadriga_src",
                        self._parameters.path_quadriga_src)
            octave.push("txs_number_antenna", self._tx_number_antennas)
            octave.push("rxs_number_antenna", self._rx_number_antennas)

            octave.push("rx_antenna_kind", self._parameters.antenna_kind)
            octave.push("number_tx", self._number_tx)
            octave.push("number_rx", self._number_rx)
            octave.push("tracks_speed", self._tracks_speed)
            octave.push("tracks_length", self._tracks_length)
            octave.push("tracks_angle", self._tracks_angle)
            octave.push("seed", self._seed)

    def _update_modem_list(
            self, modem_tx: Modem, modem_rx: Modem, update_txmodem_list: bool,
            update_rxmodem_list: bool) -> None:
        """Updates the list of modems we have"""
        if update_txmodem_list:
            self._carrier_frequency.append(modem_tx.param.carrier_frequency)
            self._sampling_rate.append(modem_tx.param.technology.sampling_rate)
            self._tx_number_antennas.append(modem_tx.param.number_of_antennas)
            self._tx_position.append(modem_tx.param.position)

        if update_rxmodem_list:
            self._rx_number_antennas.append(modem_rx.param.number_of_antennas)
            self._rx_position.append(modem_rx.param.position)

            self._tracks_length.append(modem_rx.param.track_length)
            self._tracks_angle.append(modem_rx.param.track_angle)

            vel_rx = np.linalg.norm(modem_rx.param.velocity)
            self._tracks_speed.append(vel_rx)

        self._number_tx = len(self._tx_position)
        self._number_rx = len(self._rx_position)
