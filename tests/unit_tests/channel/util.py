from unittest.mock import Mock
from typing import Dict, Any

import numpy as np


def create_mock_modem_tx(id: int) -> Mock:
    modem_tx = Mock()
    modem_tx.param.id = id
    modem_tx.param.technology.sampling_rate = 1e6
    modem_tx.param.carrier_frequency = 1e9
    modem_tx.param.number_of_antennas = 1
    modem_tx.param.position = [1, 1, 1]

    return modem_tx


def create_mock_modem_tx_from_param(id: int, param: Dict[str, Any]) -> Mock:
    modem_tx = Mock()
    modem_tx.param.id = id
    modem_tx.param.carrier_frequency = param["carrier_frequency"]
    modem_tx.param.technology.sampling_rate = param["sampling_rate"]
    modem_tx.param.number_of_antennas = param["number_of_antennas"]
    modem_tx.param.position = param["position"]

    return modem_tx


def create_mock_modem_rx(id: int) -> Mock:
    modem_rx = Mock()
    modem_rx.param.id = id
    modem_rx.param.number_of_antennas = 1
    modem_rx.param.position = [1, 1, 2]
    modem_rx.param.track_length = 2
    modem_rx.param.track_angle = 0
    modem_rx.param.velocity = np.array([0, 0, 0])

    return modem_rx


def create_mock_modem_rx_from_param(id: int, param: Dict[str, Any]) -> Mock:
    modem_rx = Mock()
    modem_rx.param.id = id
    modem_rx.param.number_of_antennas = param["number_of_antennas"]
    modem_rx.param.position = param["position"]
    modem_rx.param.track_length = param["track_length"]
    modem_rx.param.track_angle = param["track_angle"]
    modem_rx.param.velocity = param["velocity"]

    return modem_rx
