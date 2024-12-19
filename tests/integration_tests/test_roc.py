# -*- coding: utf-8 -*-

from contextlib import ExitStack
from os import path
from sys import gettrace
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from hermespy.channel import SingleTargetRadarChannel
from hermespy.hardware_loop import HardwareLoop, PhysicalScenarioDummy
from hermespy.radar import Radar, FMCW, ReceiverOperatingCharacteristic
from hermespy.simulation import SpecificIsolation, N0

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRocFromMeasurements(TestCase):
    """Test ROC estimation from previously recorded datasets"""

    def setUp(self) -> None:
        # Create temporary directory to store simulation artifacts
        self.tempdir = TemporaryDirectory()

        bandwidth = 3.072e9
        carrier_frequency = 10e9
        chirp_duration = 2e-8

        system = PhysicalScenarioDummy()
        system.noise_level = N0(1e-13)

        hardware_loop = HardwareLoop(system)
        hardware_loop.num_drops = 1
        hardware_loop.results_dir = self.tempdir.name

        device = system.new_device(carrier_frequency=carrier_frequency)
        device.isolation = SpecificIsolation(1e-8)

        radar = Radar()
        radar.waveform = FMCW(bandwidth=bandwidth, num_chirps=10, chirp_duration=chirp_duration, pulse_rep_interval=1.1 * chirp_duration)
        device.add_dsp(radar)

        channel = SingleTargetRadarChannel(target_range=(0.75, 1.25), radar_cross_section=1.0)
        system.set_channel(device, device, channel)

        with ExitStack() as stack:
            # Supress matplotlib plots
            stack.enter_context(patch("matplotlib.pyplot.figure"))

            # Suppress stdout if not in debug mode
            if gettrace() is None:
                stack.enter_context(patch("sys.stdout"))

            hardware_loop.run(overwrite=False, campaign="h1_measurements")

            channel.target_exists = False
            hardware_loop.run(overwrite=False, campaign="h0_measurements")

    def tearDown(self) -> None:
        # Clear temporary directory
        self.tempdir.cleanup()

    def test_roc_from_measurements(self) -> None:
        """Test ROC computation from measured datasets"""

        roc = ReceiverOperatingCharacteristic.From_HDF(path.join(self.tempdir.name, "drops.h5"))

        roc_data = roc.to_array()
        self.assertEqual(1, roc_data.shape[0])
        self.assertEqual(2, roc_data.shape[2])
