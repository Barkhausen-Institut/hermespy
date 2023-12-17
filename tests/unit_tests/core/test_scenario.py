# -*- coding: utf-8 -*-
"""Test HermesPy scenario description class"""

from os.path import join, exists
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import PropertyMock, MagicMock, Mock, patch

import numpy.random as rnd
from h5py import File

from hermespy.core import Drop, Scenario, ScenarioMode, Signal, SignalReceiver, SilentTransmitter, ReplayScenario
from hermespy.simulation import SimulatedDevice

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockScenario(Scenario):
    """Implementation of abstract scenario base for testing purpuses"""

    def _drop(self) -> Drop:
        # Minimal drop generation
        transmissions = self.transmit_devices()
        receptions = self.receive_devices([t.mixed_signal for t in transmissions])
        return Drop(0.0, transmissions, receptions)


class TestScenario(TestCase):
    """Test scenario base class"""

    def setUp(self) -> None:
        self.rng = rnd.default_rng(42)
        self.random_root = Mock()
        self.random_root._rng = self.rng

        self.transmitter_alpha = SilentTransmitter(10, 1.0)
        self.transmitter_beta = SilentTransmitter(10, 1.0)
        self.receiver_alpha = SignalReceiver(10, 1.0)
        self.receiver_beta = SignalReceiver(10, 1.0)

        self.device_alpha = SimulatedDevice()
        self.device_beta = SimulatedDevice()
        self.device_alpha.transmitters.add(self.transmitter_alpha)
        self.device_beta.transmitters.add(self.transmitter_beta)
        self.device_alpha.receivers.add(self.receiver_alpha)
        self.device_beta.receivers.add(self.receiver_beta)

        self.drop_duration = 1e-3
        self.scenario = MockScenario(devices=[self.device_alpha, self.device_beta])
        self.scenario.random_mother = self.random_root

    def test_arg_signature(self) -> None:
        """Arg signature class method should return correct signature"""

        expected_signature = {"seed", "devices"}
        self.assertEqual(expected_signature, Scenario._arg_signature())

    def test_add_device_validation(self) -> None:
        """Adding an already registered device should raise a ValueError"""

        with self.assertRaises(ValueError):
            self.scenario.add_device(self.device_alpha)

        with self.assertRaises(RuntimeError), patch("hermespy.core.scenario.Scenario.mode", new_callable=PropertyMock) as mode_mock:
            mode_mock.return_value = ScenarioMode.RECORD
            self.scenario.add_device(Mock())

    def test_add_device(self) -> None:
        """Adding a device should register said device to this scenario"""

        device = Mock()
        self.scenario.add_device(device)

        self.assertTrue(self.scenario.device_registered(device))

    def test_new_device(self) -> None:
        """Creating a new device should raise a RuntimeError by default"""

        with self.assertRaises(RuntimeError):
            _ = self.scenario.new_device()

    def test_device_index_validation(self) -> None:
        """Device index sould raise ValueError if device is not registered"""

        with self.assertRaises(ValueError):
            self.scenario.device_index(Mock())

    def test_device_index(self) -> None:
        """Device index should return correct index"""

        self.assertEqual(0, self.scenario.device_index(self.device_alpha))
        self.assertEqual(1, self.scenario.device_index(self.device_beta))

    def test_num_devices(self) -> None:
        """Num devices should return correct number of devices"""

        self.assertEqual(2, self.scenario.num_devices)

    def test_transmitters(self) -> None:
        """Transmitters property should return correct list of transmitters"""

        expected_transmitters = [self.transmitter_alpha, self.transmitter_beta]
        self.assertCountEqual(expected_transmitters, self.scenario.transmitters)

    def test_receivers(self) -> None:
        """Receivers property should return correct list of receivers"""

        expected_receivers = [self.receiver_alpha, self.receiver_beta]
        self.assertCountEqual(expected_receivers, self.scenario.receivers)

    def test_num_receivers(self) -> None:
        """Number of receivers property should return correct number of receivers"""

        self.assertEqual(2, self.scenario.num_receivers)

    def test_num_transmitters(self) -> None:
        """Number of transmitters property should return the currect number"""

        self.assertEqual(2, self.scenario.num_transmitters)

    def test_operators(self) -> None:
        """Receivers property should return correct list of operators"""

        expected_operators = [self.transmitter_alpha, self.transmitter_beta, self.receiver_alpha, self.receiver_beta]
        self.assertCountEqual(expected_operators, self.scenario.operators)

    def test_num_operators(self) -> None:
        """Number of operators property should return correct number of operators"""

        self.assertEqual(4, self.scenario.num_operators)

    def test_drop_duration_setget(self) -> None:
        """The drop duration property getter should return the setter argument,"""

        drop_duration = 12345
        self.scenario.drop_duration = drop_duration

        self.assertEqual(drop_duration, self.scenario.drop_duration)

    def test_drop_duration_validation(self) -> None:
        """The drop duration property setter should raise a ValueError on negative arguments"""

        with self.assertRaises(ValueError):
            self.scenario.drop_duration = -1

        try:
            self.scenario.drop_duration = 0.0

        except ValueError:
            self.fail("Setting a drop duration of zero should not result in an error throw")

        with self.assertRaises(RuntimeError), patch("hermespy.core.scenario.Scenario.mode", new_callable=PropertyMock) as mode_mock:
            mode_mock.return_value = ScenarioMode.RECORD
            self.scenario.drop_duration = 1.0

    def test_drop_duration_computation(self) -> None:
        """If the drop duration is set to zero,
        the property getter should return the maximum frame duration as drop duration"""

        max_frame_duration = 10.0  # Results from the setUp transmit mock
        self.scenario.drop_duration = 0.0

        self.assertEqual(max_frame_duration, self.scenario.drop_duration)

    def test_campaign_validation(self) -> None:
        """The campaign property setter should raise a ValueError on invalid calls"""

        with patch("hermespy.core.scenario.Scenario.mode", new_callable=PropertyMock) as mode_mock, patch("hermespy.core.scenario.Scenario._Scenario__campaign_exists") as campaign_exists_mock:
            mode_mock.return_value = ScenarioMode.REPLAY
            campaign_exists_mock.return_value = False

            with self.assertRaises(ValueError):
                self.scenario.campaign = "test"

    def test_campaign_setget(self) -> None:
        """The campaign property getter should return the setter argument"""

        self.scenario.campaign = "test1"
        self.assertEqual("test1", self.scenario.campaign)

        # Second call for coverage
        self.scenario.campaign = "test2"
        self.assertEqual("test2", self.scenario.campaign)

        with patch("hermespy.core.scenario.Scenario.mode", new_callable=PropertyMock) as mode_mock, patch("hermespy.core.scenario.Scenario._Scenario__campaign_exists") as campaign_exists_mock:
            mode_mock.return_value = ScenarioMode.REPLAY
            campaign_exists_mock.return_value = True

            self.scenario.campaign = "test3"
            self.assertEqual("test3", self.scenario.campaign)

            mode_mock.return_value = ScenarioMode.RECORD
            campaign_exists_mock.return_value = False

            with patch.object(self.scenario, "_Scenario__file", new_callable=MagicMock) as file_mock:
                self.scenario.campaign = "test4"
                self.assertEqual("test4", self.scenario.campaign)

    def test_record_validation(self) -> None:
        """Record should raise a RuntimeError if the scenario is already recording"""

        with patch("hermespy.core.scenario.Scenario.mode", new_callable=PropertyMock) as mode_mock:
            mode_mock.return_value = ScenarioMode.REPLAY

            with self.assertRaises(RuntimeError):
                self.scenario.record("test.hdf")

    def test_record_overwrite(self) -> None:
        """Record should replace existing file if overwrite flag is enabled"""

        with TemporaryDirectory() as tmp_dir:
            file_path = join(tmp_dir, "test.hdf")

            self.scenario.record(file_path)
            self.scenario.stop()

            self.scenario.record(file_path, overwrite=True)
            self.scenario.stop()

            self.assertTrue(exists(file_path))

    def test_record_custom_state(self) -> None:
        """Record should use custom state if provided"""

        with TemporaryDirectory() as tmp_dir:
            file_path = join(tmp_dir, "test.hdf")
            self.scenario.record(file_path, campaign="test", state=self.scenario)
            self.scenario.stop()

        self.assertEqual(self.scenario.campaign, "test")

    def test_record_replay(self) -> None:
        """Record and replay should return the same drop"""

        with TemporaryDirectory() as tmp_dir:
            campagin = "abrc"
            self.scenario.drop_duration = 1e-3

            file_path = join(tmp_dir, "test.hdf")

            self.scenario.record(file_path, campaign=campagin)
            recorded_drop = self.scenario.drop()

            self.scenario.replay(campaign=campagin)
            directly_replayed_drop = self.scenario.drop()

            self.scenario.stop()

            replayed_scenario = MockScenario.Replay(file_path, campaign=campagin)
            replayed_drop = replayed_scenario.drop()
            replayed_scenario.stop()

        self.assertEqual(campagin, replayed_scenario.campaign)
        self.assertEqual(recorded_drop.timestamp, replayed_drop.timestamp)
        self.assertEqual(recorded_drop.num_device_receptions, replayed_drop.num_device_receptions)
        self.assertEqual(recorded_drop.num_device_transmissions, replayed_drop.num_device_receptions)

    def test_replay_validation(self) -> None:
        """Replay should raise a ValueError if the requested campaign does not exist"""

        with self.assertRaises(ValueError):
            self.scenario.replay()

        with TemporaryDirectory() as tmp_dir:
            campaign = "abrc"
            self.scenario.drop_duration = 1e-3

            file_path = join(tmp_dir, "test.hdf")
            self.scenario.record(file_path, campaign=campaign)
            _ = self.scenario.drop()
            self.scenario.stop()

            with patch("hermespy.core.scenario.Scenario._Scenario__campaign_exists") as campaign_exists_mock:
                campaign_exists_mock.return_value = False

                with self.assertRaises(ValueError):
                    self.scenario.replay(file_path, campaign=campaign)

    def test_replay_from_file(self) -> None:
        """Replay from file should return the same drop"""

        with TemporaryDirectory() as tmp_dir:
            campagin = "abrc"
            self.scenario.drop_duration = 1e-3

            file_path = join(tmp_dir, "test.hdf")
            self.scenario.record(file_path, campaign=campagin)
            recorded_drop = self.scenario.drop()
            self.scenario.stop()

            file = File(file_path, "r")
            self.scenario.replay(file, campaign=campagin)
            replayed_drop = self.scenario.drop()
            self.scenario.stop()

        self.assertEqual(recorded_drop.timestamp, replayed_drop.timestamp)
        self.assertEqual(recorded_drop.num_device_receptions, replayed_drop.num_device_receptions)
        self.assertEqual(recorded_drop.num_device_transmissions, replayed_drop.num_device_receptions)

    def test_transmit_operators(self) -> None:
        """Transmit operators should return the correct list of operators"""

        mock_alpha = MagicMock()
        mock_beta = MagicMock()
        self.device_alpha.transmitters.add(mock_alpha)
        self.device_beta.transmitters.add(mock_beta)

        transmissions = self.scenario.transmit_operators()

        self.assertEqual(2, len(transmissions))
        mock_alpha.transmit.assert_called_once()
        mock_beta.transmit.assert_called_once()

    def test_generate_outputs_validation(self) -> None:
        """Generate outputs should raise a ValueError for an invalid transmisison list"""

        transmissions = self.scenario.transmit_operators()

        mock_device = MagicMock()
        self.scenario.add_device(mock_device)

        with self.assertRaises(ValueError):
            _ = self.scenario.generate_outputs(transmissions)

    def test_generate_outputs(self) -> None:
        """Generate outputs should return the correct list of outputs"""

        mock_device = MagicMock()
        self.scenario.add_device(mock_device)

        outputs = self.scenario.generate_outputs()

        self.assertEqual(3, len(outputs))
        mock_device.generate_output.assert_called_once()

    def test_process_inputs_validation(self) -> None:
        """Process inputs should raise a ValueError for an invalid impinging signals list"""

        with self.assertRaises(ValueError):
            _ = self.scenario.process_inputs([])

    def test_process_inputs(self) -> None:
        impinging_signals = [Signal(self.rng.random((1, 10)), 1.0, 0.0) for _ in range(self.scenario.num_devices)]
        processed_inputs = self.scenario.process_inputs(impinging_signals)

        self.assertEqual(2, len(processed_inputs))

    def test_receive_operators_validation(self) -> None:
        """Receive operators should raise a ValueError for an invalid operator inputs list"""

        with self.assertRaises(ValueError):
            _ = self.scenario.receive_operators([])

    def test_receive_operators(self) -> None:
        impinging_signals = [Signal(self.rng.random((1, 10)), 1.0, 0.0) for _ in range(self.scenario.num_devices)]
        processed_inputs = self.scenario.process_inputs(impinging_signals)
        receptions = self.scenario.receive_operators(processed_inputs)

        self.assertEqual(2, len(receptions))

    def test_num_drop_devices(self) -> None:
        """Num drop devices should return the correct number of devices"""

        with TemporaryDirectory() as tmp_dir:
            self.scenario.drop_duration = 1e-3

            file_path = join(tmp_dir, "test.hdf")
            self.scenario.record(file_path)
            _ = self.scenario.drop()
            self.assertEqual(1, self.scenario.num_drops)
            self.scenario.stop()

            replayed_scenario = MockScenario.Replay(file_path)
            _ = replayed_scenario.drop()
            self.assertEqual(1, replayed_scenario.num_drops)
            replayed_scenario.stop()

        self.assertIsNone(self.scenario.num_drops)


class TestReplayScenario(TestCase):
    """Test replay scenario class"""

    def setUp(self) -> None:
        self.scenario = ReplayScenario()

    def test_drop_validation(self) -> None:
        """Dropping a replay scenario should raise a RuntimeError"""

        with self.assertRaises(RuntimeError):
            self.scenario.drop()
