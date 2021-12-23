import unittest

from hermespy.core.factory import Factory

def create_scenario_stream_header() -> str:
    return """
!<Scenario>

sampling_rate: 2e6
"""

def create_random_modem_yaml_str(modem_type: str) -> str:
    if modem_type.upper() not in ["TRANSMITTER", "RECEIVER"]:
        raise ValueError("Modem type not supported")

    return f"""
  - {modem_type}
    carrier_frequency: 1e9
    position: [0, 0, 0]
    WaveformChirpFsk:
        chirp_bandwidth: 5e5
        chirp_duration: 512e-6
        freq_difference: 1953.125
        num_data_chirps: 12
        modulation_order: 32
"""

def create_section_yaml_str(section: str) -> str:
    return f"""
{section}:"""

def create_channel_yaml_str(tx: int, rx: int) -> str:
    return f"""
  - Channel {tx} {rx}
    active: true

"""
def create_sync_offset_yaml_str(low: float, high: float) -> str:

    return f"""
    sync_offset_low: {low}
    sync_offset_high: {high}
"""

class TestChannelTimeoffsetScenarioCreation(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario_str = create_scenario_stream_header()
        self.scenario_str += create_section_yaml_str("Modems")
        self.scenario_str += create_random_modem_yaml_str("Transmitter")
        self.scenario_str += create_random_modem_yaml_str("Receiver")

        self.scenario_str += create_section_yaml_str("Channels")
        self.scenario_str += create_channel_yaml_str(0, 0)
        self.factory = Factory()

    def test_setup_single_offset_correct_initialization_with_correct_values(self) -> None:
        LOW = 1
        HIGH = 5
        self.scenario_str += create_sync_offset_yaml_str(LOW, HIGH)
        scenario = self.factory.from_str(self.scenario_str)

        self.assertEqual(scenario[0].channels[0, 0].sync_offset_low, LOW)
        self.assertEqual(scenario[0].channels[0, 0].sync_offset_high, HIGH)

    def test_no_parameters_result_in_default_values(self) -> None:
        scenario = self.factory.from_str(self.scenario_str)

        self.assertEqual(scenario[0].channels[0, 0].sync_offset_low, 0)
        self.assertEqual(scenario[0].channels[0, 0].sync_offset_high, 0)

    def test_exception_raised_if_high_smaller_than_low(self) -> None:
        LOW = 2
        HIGH = 1

        self.scenario_str += create_sync_offset_yaml_str(LOW, HIGH)
        with self.assertRaises(ValueError):
            scenario = self.factory.from_str(self.scenario_str)

    def test_exception_raised_if_low_smaller_than_zero(self) -> None:
        LOW = -1
        HIGH = 0

        self.scenario_str += create_sync_offset_yaml_str(LOW, HIGH)
        with self.assertRaises(ValueError):
            scenario = self.factory.from_str(self.scenario_str)

    def test_exception_raised_if_high_smaller_than_zero(self) -> None:
        LOW = -1
        HIGH = -5

        self.scenario_str += create_sync_offset_yaml_str(LOW, HIGH)
        with self.assertRaises(ValueError):
            scenario = self.factory.from_str(self.scenario_str)

    def test_multiple_channels_creation_all_sync_offsets_defined(self) -> None:
        sync_offsets = {
            'ch0_0': {'LOW': 0, 'HIGH': 3},
            'ch1_0': {'LOW': 2, 'HIGH': 5},
            'ch0_1': {'LOW': 1, 'HIGH': 4},
            'ch1_1': {'LOW': 5, 'HIGH': 10}
        }
        s = create_scenario_stream_header()
        s += create_section_yaml_str("Modems")
        s += create_random_modem_yaml_str("Transmitter")
        s += create_random_modem_yaml_str("Transmitter")

        s += create_random_modem_yaml_str("Receiver")
        s += create_random_modem_yaml_str("Receiver")

        s += create_section_yaml_str("Channels")
        s += create_channel_yaml_str(0, 0)
        s += create_sync_offset_yaml_str(low=sync_offsets['ch0_0']['LOW'], high=sync_offsets['ch0_0']['HIGH'])
        s += create_channel_yaml_str(1, 0)
        s += create_sync_offset_yaml_str(low=sync_offsets['ch1_0']['LOW'], high=sync_offsets['ch1_0']['HIGH'])
        s += create_channel_yaml_str(0, 1)
        s += create_sync_offset_yaml_str(low=sync_offsets['ch0_1']['LOW'], high=sync_offsets['ch0_1']['HIGH'])
        s += create_channel_yaml_str(1, 1)
        s += create_sync_offset_yaml_str(low=sync_offsets['ch1_1']['LOW'], high=sync_offsets['ch1_1']['HIGH'])

        scenario = self.factory.from_str(s)
        ch = scenario[0].channels
        self.assertEqual(ch[0, 0].sync_offset_low, sync_offsets['ch0_0']['LOW'])
        self.assertEqual(ch[0, 0].sync_offset_high, sync_offsets['ch0_0']['HIGH'])

        self.assertEqual(ch[0, 1].sync_offset_low, sync_offsets['ch0_1']['LOW'])
        self.assertEqual(ch[0, 1].sync_offset_high, sync_offsets['ch0_1']['HIGH'])

        self.assertEqual(ch[1, 0].sync_offset_low, sync_offsets['ch1_0']['LOW'])
        self.assertEqual(ch[1, 0].sync_offset_high, sync_offsets['ch1_0']['HIGH'])

        self.assertEqual(ch[1, 1].sync_offset_low, sync_offsets['ch1_1']['LOW'])
        self.assertEqual(ch[1, 1].sync_offset_high, sync_offsets['ch1_1']['HIGH'])
