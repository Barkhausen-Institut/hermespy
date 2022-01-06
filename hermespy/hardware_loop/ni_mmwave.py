# -*- coding: utf-8 -*-
"""
===========================================
National Instruments MmWave Device Binding
===========================================
"""

import numpy as np
from datetime import datetime

import mmw.mmw as mmw

from hermespy.core.signal_model import Signal
from .physical_device import PhysicalDevice


class NiMmWaveDevice(PhysicalDevice):

    scale_reception: bool
    """Scale received signal power."""

    __driver: mmw.ni_mmw
    __sampling_rate: float
    __carrier_frequency: float
    __waveform_counter: int
    __burst_prefix: str
    __tx_gain_db: float
    __rx_gain_db: float
    __time_buffer: float

    def __init__(self,
                 host: str,
                 port: int = 5555,
                 timeout=40000,
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

        # Initialize waveform counter
        self.__waveform_counter = 0
        self.__burst_prefix = datetime.now().strftime("%Y%m%d%H%M%S_")

        # Initialize MmWave driver
        self.__driver = mmw.ni_mmw(host=host, port=port)

        # Initialize hardware
        self.__assert_cmd(self.__driver.initialize_hw(mmw.const.opmodes.RF, timeout=timeout))
        self.__assert_cmd(self.__driver.enable_LO_sync(True))
        self.__assert_cmd(self.__driver.trigger_sync_enable(True))

        # Initialize base class
        PhysicalDevice.__init__(self, *args, **kwargs)

        self.__tx_gain_db = 0.
        self.__rx_gain_db = 0.
        self.__carrier_frequency = 0.
        self.__scale_reception = True
        self.time_buffer = 0.

        # Configure default parameters
        self.sampling_rate = 500000
        self.carrier_frequency = 75e9

    @property
    def carrier_frequency(self) -> float:

        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> None:

        self.__assert_cmd(self.__driver.configure_rf(value, self.__tx_gain_db, mmw.const.ports.tx))
        self.__assert_cmd(self.__driver.configure_rf(value, self.__rx_gain_db, mmw.const.ports.rx))

        self.__carrier_frequency = value

    @property
    def tx_gain_db(self) -> float:
        """Transmit power gain.

        Returns:
            gain (float):
                Gain in dB.
        """

        return self.__tx_gain_db

    @tx_gain_db.setter
    def tx_gain_db(self, value: float) -> None:
        
        self.__assert_cmd(self.__driver.configure_rf(self.__carrier_frequency, value, mmw.const.ports.tx))
        self.__tx_gain_db = value

    @property
    def rx_gain_db(self) -> float:
        """Receive power gain.

        Returns:
            gain (float):
                Gain in dB.
        """

        return self.__rx_gain_db

    @rx_gain_db.setter
    def rx_gain_db(self, value: float) -> None:

        self.__assert_cmd(self.__driver.configure_rf(self.__carrier_frequency, value, mmw.const.ports.rx))
        self.__rx_gain_db = value

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
    def time_buffer(self) -> float:
        """Signal sample offset buffer.

        Returns:
            buffer (float):
                Offset buffer in seconds.
        """

        return self.__time_buffer

    @time_buffer.setter
    def time_buffer(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Time buffer must be greater or equal to zero")

        self.__time_buffer = value

    @property
    def num_antennas(self) -> int:

        return 1

    def trigger(self) -> None:

        # Compute signal to be transmitted
        transmitted_signal = self.transmit()
        transmitted_samples = transmitted_signal.to_interleaved(np.int16)

        if self.time_buffer > 0.:

            num_buffer_samples = int(self.time_buffer * self.sampling_rate)
            transmitted_samples = np.append(np.zeros(2 * num_buffer_samples, dtype=np.int16), transmitted_samples)

        # Burst unique identifier generation
        burst_uid = self.__burst_prefix + str(self.__waveform_counter)
        self.__waveform_counter += 1

        # Upload transmit samples to instrument memory
        self.__assert_cmd(self.__driver.write_tx(burst_uid, transmitted_samples))

        # Configure acquisition parameters
        acquisition_length = transmitted_signal.duration + 2 * self.time_buffer
        self.__assert_cmd(self.__driver.start([burst_uid], acquisition_length, 20000))# , acquisition_length, 60000))

        # Trigger hardware
        self.__assert_cmd(self.__driver.send_trigger(burstmode=mmw.const.burst_mode.burst))

        # Download received samples
        response, interleaved_samples = self.__driver.fetch()
        self.__assert_cmd(response)

        # Recover signal model from interleaved complex samples
        received_signal = Signal.from_interleaved(interleaved_samples, sampling_rate=self.__sampling_rate)

        # Scale the signal to unit max amplitude
        if self.__scale_reception:

            max_amplitude = np.max(abs(received_signal.samples))

            if max_amplitude > 0.:
                received_signal.samples = received_signal.samples / max_amplitude

        # Cache the received signal
        self.receive(received_signal)

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


class NiMmWaveDualDevice(PhysicalDevice):

    __driver_A: mmw.ni_mmw
    __driver_B: mmw.ni_mmw
    __sampling_rate: float
    __carrier_frequency: float
    __waveform_counter: int

    def __init__(self,
                 host_a: str,
                 host_b: str,
                 port_a: int = 5555,
                 port_b: int = 5555,
                 timeout=40000,
                 carrier_frequency: float = 75e9,
                 *args, **kwargs) -> None:
        """
        Args:

            host_a (str):
                Host of address the USRP.
                For example '127.0.0.1' or 'device.tld'.


            host_b (str):
                Host of address the USRP.
                For example '127.0.0.1' or 'device.tld'.

            port_a (int, optional):
                Listening port of `host_A`.


            port_b (int, optional):
                Listening port of `host_B`.

            timeout (int, optional):
                Network connection timeout.

            carrier_frequency (float, optional):
                Center frequency of the radio-frequency band signal.
                75 GHz by default.

            *args:
                Device base class initialization parameters.

            **kwargs:
                Device base class initialization parameters.

        Raises:

            RuntimeError:
                If device initialization fails.
        """

        # Initialize counter
        self.__waveform_counter = 0

        # Initialize MmWave driver
        self.__driver_A = mmw.ni_mmw(host=host_a, port=port_a)
        self.__driver_B = mmw.ni_mmw(host=host_b, port=port_b)

        # Sending the command a reading response
#        _ = self.__driver_A._send_instr_command("reset_hw", {"no_param": ""})
#        _ = self.__driver_B._send_instr_command("reset_hw", {"no_param": ""})

        # Initialize hardware
        self.__assert_cmd(self.__driver_A.initialize_hw(mmw.const.opmodes.RF, timeout=timeout))
        self.__assert_cmd(self.__driver_B.initialize_hw(mmw.const.opmodes.RF, timeout=timeout))
        self.__assert_cmd(self.__driver_A.enable_LO_sync(True))
        self.__assert_cmd(self.__driver_B.enable_LO_sync(True))
        self.__assert_cmd(self.__driver_A.trigger_sync_enable(True))
        self.__assert_cmd(self.__driver_B.trigger_sync_enable(True))

        # Initialize base class
        PhysicalDevice.__init__(self, *args, **kwargs)

        # Configure default parameters
        self.carrier_frequency = carrier_frequency
        self.sampling_rate = 500000

    def __del__(self):
        """Close network connections on object deletion."""

        self.__driver_A.shutdown()
        self.__driver_B.shutdown()

    @property
    def carrier_frequency(self) -> float:

        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> None:

        self.__assert_cmd(self.__driver_A.configure_rf(value, 0., mmw.const.ports.tx))
        self.__assert_cmd(self.__driver_A.configure_rf(value, 0., mmw.const.ports.rx))
        self.__assert_cmd(self.__driver_B.configure_rf(value, 0., mmw.const.ports.tx))
        self.__assert_cmd(self.__driver_B.configure_rf(value, 0., mmw.const.ports.rx))

        self.__carrier_frequency = value

    @property
    def sampling_rate(self) -> float:

        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:

        if value <= 0.:
            raise ValueError("Sampling rate must be greater than zero")

        self.__assert_cmd(self.__driver_A.configure_fs(value, mmw.const.ports.rx))
        self.__assert_cmd(self.__driver_A.configure_fs(value, mmw.const.ports.tx))
        self.__assert_cmd(self.__driver_B.configure_fs(value, mmw.const.ports.rx))
        self.__assert_cmd(self.__driver_B.configure_fs(value, mmw.const.ports.tx))

        self.__sampling_rate = value

    @property
    def num_antennas(self) -> int:

        return 1

    def trigger(self) -> None:

        # Compute signal to be transmitted
        transmitted_signal = self.transmit()

        transmitted_samples = transmitted_signal.to_interleaved(np.int16)
        burst_uid = f"burst_{self.__waveform_counter}"

        # Upload transmit samples to instrument memory
        self.__assert_cmd(self.__driver_A.write_tx(burst_uid, transmitted_samples))
        self.__assert_cmd(self.__driver_B.write_tx(burst_uid, transmitted_samples))

        # Configure acquisition parameters
        acquisition_length = transmitted_signal.duration    # ToDo: Find a better way to handle this
        self.__assert_cmd(self.__driver_A.start([burst_uid], acquisition_length, 20000))# , acquisition_length, 60000))
        self.__assert_cmd(self.__driver_B.start([burst_uid], acquisition_length, 20000))# , acquisition_length, 60000))

        # Trigger hardware
        self.__assert_cmd(self.__driver_A.send_trigger(burstmode=mmw.const.burst_mode.burst))

        # Update burst counter
        self.__waveform_counter += 1

        # Download received samples
        response, interleaved_samples_a = self.__driver_A.fetch()
        self.__assert_cmd(response)
        response, interleaved_samples_b = self.__driver_B.fetch()
        self.__assert_cmd(response)

        self.receive(Signal.from_interleaved(interleaved_samples_b, sampling_rate=self.__sampling_rate))

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
