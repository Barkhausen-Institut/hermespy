# -*- coding: utf-8 -*-
"""
===========================================
National Instruments MmWave Device Binding
===========================================
"""

import mmw.mmw as mmw

from .physical_device import PhysicalDevice


class NiMmWaveDevice(PhysicalDevice):

    __driver: mmw.ni_mmw
    __sampling_rate: float
    __carrier_frequency: float

    def __init__(self,
                 host: str,
                 port: int = 5000,
                 timeout=10000,
                 *args, **kwargs) -> None:
        """
        Args:

            host (str):
                Host of address the USRP.
                For example '127.0.0.1' or 'device.tld'.

            port (int, optional):
                Listening port of `host`.

            timeout (int, optional):
                Network connection timeout.

            *args:
                Device base class initialization parameters.

            **kwargs:
                Device base class initialization parameters.

        Raises:

            RuntimeError:
                If device initialization fails.
        """

        # Initialize MmWave driver
        self.__driver = mmw.ni_mmw(host=host, port=port)

        # Initialize hardware
        self.__assert_cmd(self.__driver.initialize_hw(mmw.const.opmodes.RF, timeout=timeout))
        self.__assert_cmd(self.__driver.enable_LO_sync(True))
        self.__assert_cmd(self.__driver.trigger_sync_enable(True))

        # Initialize base class
        PhysicalDevice.__init__(self, *args, carrier_frequency=75e9, **kwargs)

        # Configure default parameters
        self.sampling_rate = 500000

    @property
    def carrier_frequency(self) -> float:

        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> None:

        self.__assert_cmd(self.__driver.configure_rf(value, -10, mmw.const.ports.tx))
        self.__assert_cmd(self.__driver.configure_rf(value, 10, mmw.const.ports.rx))

        self.__carrier_frequency = value

    @property
    def sampling_rate(self) -> float:

        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:

        if value <= 0.:
            raise ValueError("Sampling rate must be greater than zero")

        self.__assert_cmd(self.__driver.configure_fs(value, mmw.const.ports.rx))
        self.__assert_cmd(self.__driver.configure_fs(value, mmw.const.ports.tx))

        self.__sampling_rate = value

    @property
    def num_antennas(self) -> int:

        return 1

    def trigger(self) -> None:

        # Compute signal to be transmitted
        transmitted_signal = self.transmit()

        # Upload transmit samples to instrument memory
        self.__assert_cmd(self.__driver.write_tx("waveform", transmitted_signal.samples))

        # Configure acquisition parameters
        acquisition_length = transmitted_signal.duration    # ToDo: Find a better way to handle this
        self.__assert_cmd(self.__driver.start(["waveform"], acquisition_length, 5000))

        # Trigger hardware
        self.__assert_cmd(self.__driver.send_trigger(burstmode=mmw.const.burst_mode.burst))

        # Download received samples
        response, data = self.__driver.fetch()
        self.__assert_cmd(response)

        self.receive(data)

    @staticmethod
    def __assert_cmd(result: dict) -> None:
        """Check if a command result from a driver query indicates failure.

        Raises:
            RuntimeError: On failed commands.
        """

        if isinstance(result, tuple):
            result = result[0]

        if 'Errcode' not in result:
            raise RuntimeError("mmWave driver query failed for unknown reason")

        if result['Errcode'] != 'OK':

            try:

                raise RuntimeError("mmWave driver error: " + result['Parameters']['generic_info'])

            except LookupError:

                raise RuntimeError("mmWave driver query failed for unknown reason")
