# -*- coding: utf-8 -*-
"""
===========================================
National Instruments MmWave Device Binding
===========================================
"""

import mmw.mmw as mmw

from hermespy.signal import Signal
from .physical_device import PhysicalDevice

class NiMmWaveDevice(PhysicalDevice):

    __driver: mmw.ni_mmw
    __sampling_rate: float
    __carrier_frequency: float

    def __init__(self,
                 host: str,
                 port: int = 5558,
                 *args, **kwargs) -> None:
        """
        Args:

            host (str):
                Host of the USRP.
                For example '127.0.0.1' or 'device.tld'.

            port (int, optional):
                Listening port of `host`.

            *args:
                Device base class initialization parameters.

            **kwargs:
                Device base class initialization parameters.

        Raises:

            RuntimeError:
                If device initialization fails.
        """

        # Initialize base class
        PhysicalDevice.__init__(self, *args, **kwargs)

        # Initialize MmWave driver
        self.__driver = mmw.ni_mmw(host=host, port=port)

        # Initialize hardware
        if self.__driver.initialize_hw(mmw.const.opmodes.RF) != 0:
            raise RuntimeError("NI mmWave hardware init failed")

        # Configure default parameters
        self.sampling_rate = 3.072e9
        self.carrier_frequency = 75e9

    @property
    def carrier_frequency(self) -> float:

        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> None:

        if self.__driver.configure_rf(value, -10, mmw.const.ports.tx) != 0: # gain = -10dB
            raise RuntimeError("Error configuring transmit carrier frequency")

        if self.__driver.configure_rf(value, 10, mmw.const.ports.rx) != 0:
            raise RuntimeError("Error configuring receive carrier frequency")

        self.__carrier_frequency = value

    @property
    def sampling_rate(self) -> float:

        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None

        if value <= 0.:
            raise ValueError("Sampling rate must be greater than zero")

        if self.__driver.configure_fs(value, mmw.const.ports.rx) != 0:
            raise RuntimeError("Error configuring transmit sampling rate")

        if self.__driver.configure_fs(value, mmw.const.ports.tx) != 0:
            raise RuntimeError("Error configuring receive sampling rate")

        self.__sampling_rate = value

    @property
    def num_antennas(self) -> int:

        return 1

    def trigger(self) -> None:

        # Compute signal to be transmitted
        transmitted_signal = self.transmit()

        # Upload transmit samples to instrument memory
        self.__driver.write_tx("waveform", transmitted_signal.samples)

        # Configure acquisition parameters
        self.__driver.start(["waveform"], 8e-6, 5000)

        # Trigger hardware
        self.__driver.send_trigger(burstmode=mmw.const.burst_mode.burst)

        # Download received samples
        response, data = self.__driver.fetch()

        received_signal = Signal(samples=data, sampling_rate=self.__sampling_rate)
        self.receive(data)
