# -*- coding: utf-8 -*-

import logging
import os.path as path
from unittest import TestCase

import ray as ray

from unit_tests.utils import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestDocumentationExamples(TestCase):
    def setUp(self) -> None:
        self.test_context = SimulationTestContext()
        self.base_path = path.abspath(path.join(path.dirname(path.abspath(__file__)), "..", "..", "..", "docssource", "scripts", "examples"))

    @classmethod
    def setUpClass(cls) -> None:
        ray.init(local_mode=True, num_cpus=1, ignore_reinit_error=True, logging_level=logging.ERROR)

    @classmethod
    def tearDownClass(cls) -> None:
        # Shut down ray
        ray.shutdown()

    def __run_example(self, script: str) -> None:
        """Run a python script from the documentation's examples directory.

        Args:

            script (str):
                Path to the python script relative to the examples directory.
        """

        script_path = path.join(self.base_path, script)

        with self.test_context:
            exec(open(script_path).read())
            
    def test_beamforming_nullsteeringbeamformer_radiation_pattern(self) -> None:
        """Test example snippet for nullsteeringbeamformer radition pattern"""

        self.__run_example("beamforming_nullsteeringbeamformer_radiation_pattern.py")

    def test_beamforming_nullsteeringbeamformer(self) -> None:
        """Test example snippet for nullsteeringbeamformer"""

        self.__run_example("beamforming_nullsteeringbeamformer.py")

    def test_channel_cdl_indoor_factory(self) -> None:
        """Test example snippet for indoor channel factory with LOS"""

        self.__run_example("channel_cdl_indoor_factory.py")

    def test_channel_cdl_indoor_offices(self) -> None:
        """Test example snippet for indoor office channel with LOS"""

        self.__run_example("channel_cdl_indoor_office.py")

    def test_channel_cdl_rural_macrocells(self) -> None:
        """Test example snippet for rural macrocells channel with LOS"""

        self.__run_example("channel_cdl_rural_macrocells.py")

    def test_channel_cdl_street_canyon(self) -> None:
        """Test example snippet for street canyon channel with LOS"""

        self.__run_example("channel_cdl_street_canyon.py")

    def test_channel_cdl_urban_macrocells(self) -> None:
        """Test example snippet for urban macrocells channel with LOS"""

        self.__run_example("channel_cdl_urban_macrocells.py")

    def test_channel_fading_TDL(self) -> None:
        """Test example snippet for 5G TDL multipath fading channel"""

        self.__run_example("channel_fading_tdl.py")

    def test_channel_fading_fading(self) -> None:
        """Test example snippet for multipath fading channel"""

        self.__run_example("channel_fading_fading.py")

    def test_channel_fading_cost259(self) -> None:
        """Test example snippet for COST 259 multipath fading channel"""

        self.__run_example("channel_fading_cost259.py")

    def test_channel_fading_exponential(self) -> None:
        """Test example snippet for exponential multipath fading channel"""

        self.__run_example("channel_fading_exponential.py")

    def test_jcas_ofdm_radar(self) -> None:
        """Test example snippet for OFDM radar"""

        self.__run_example("jcas_ofdm_radar.py")

    def test_channel_radar_multi(self) -> None:
        """Test example snippet for multi-target radar channel"""

        self.__run_example("channel_radar_multi.py")

    def test_channel_radar_single(self) -> None:
        """Test example snippet for single-target radar channel"""

        self.__run_example("channel_radar_single.py")

    def test_channel_delay_random(self) -> None:
        """Test example snippet for random delay channel"""

        self.__run_example("channel_delay_random.py")

    def test_channel_delay_spatial(self) -> None:
        """Test example snippet for spatial delay channel"""

        self.__run_example("channel_delay_spatial.py")

    def test_channel(self) -> None:
        """Test example snippet for channel"""

        self.__run_example("channel.py")

    def test_modem_DuplexModem(self) -> None:
        """Test example snippet for duplex modem"""

        self.__run_example("modem_DuplexModem.py")

    def test_modem_evaluators_ber(self) -> None:
        """Test example snippet for bit error rate evaluation"""

        self.__run_example('modem_evaluators_ber.py')

    def test_modem_evaluators_bler(self) -> None:
        """Test example snippet for block error rate evaluation"""

        self.__run_example('modem_evaluators_bler.py')

    def test_modem_evaluators_evm(self) -> None:
        """Test example snippet for error vector magnitude evaluation"""

        self.__run_example('modem_evaluators_evm.py')

    def test_modem_evaluators_fer(self) -> None:
        """Test example snippet for frame error rate evaluation"""

        self.__run_example('modem_evaluators_fer.py')

    def test_modem_evaluators_throughput(self) -> None:
        """Test example snippet for throughput evaluation"""

        self.__run_example('modem_evaluators_throughput.py')

    def test_modem_precoding_alamouti(self) -> None:
        """Test example snippet for Alamouti precoding"""

        self.__run_example('modem_precoding_alamouti.py')

    def test_modem_precoding_dft(self) -> None:
        """Test example snippet for DFT precoding"""

        self.__run_example('modem_precoding_dft.py')

    def test_modem_precoding_ganesan(self) -> None:
        """Test example snippet for Ganesan precoding"""

        self.__run_example('modem_precoding_ganesan.py')

    def test_modem_precoding_mrc(self) -> None:
        """Test example snippet for MRC precoding"""

        self.__run_example('modem_precoding_mrc.py')

    def test_modem_precoding_precoding(self) -> None:
        """Test example snippet for precoding"""

        self.__run_example('modem_precoding_precoding.py')

    def test_modem_precoding_sc(self) -> None:
        """Test example snippet for SC precoding"""

        self.__run_example('modem_precoding_sc.py')

    def test_modem_ReceivingModem(self) -> None:
        """Test example snippet for receiving modem"""

        self.__run_example("modem_ReceivingModem.py")

    def test_modem_SimplexLink(self) -> None:
        """Test example snippet for simplex link"""

        self.__run_example("modem_SimplexLink.py")

    def test_modem_TransmittingModem(self) -> None:
        """Test example snippet for transmitting modem"""

        self.__run_example("modem_TransmittingModem.py")

    def test_modem_waveforms_cfsk(self) -> None:
        """Test example snippet for CFSK waveforms"""

        self.__run_example("modem_waveforms_cfsk.py")

    def test_modem_waveforms_fmcw(self) -> None:
        """Test example snippet for FMCW waveforms"""

        self.__run_example("modem_waveforms_fmcw.py")

    def test_modem_waveforms_rc(self) -> None:
        """Test example snippet for RC waveforms"""

        self.__run_example("modem_waveforms_rc.py")

    def test_modem_waveforms_rect(self) -> None:
        """Test example snippet for rect waveforms"""

        self.__run_example("modem_waveforms_rect.py")

    def test_modem_waveforms_rrc(self) -> None:
        """Test example snippet for RRC waveforms"""

        self.__run_example("modem_waveforms_rrc.py")

    def test_modem_waveforms_orthogonal(self) -> None:
        """Test example snippet for orthogonal waveforms"""

        self.__run_example("modem_waveforms_orthogonal.py")

    def test_modem_waveforms_ofdm(self) -> None:
        """Test example snippet for OFDM waveforms"""

        self.__run_example("modem_waveforms_ofdm.py")

    def test_modem_waveforms_ocdm(self) -> None:
        """Test example snippet for OCDM waveforms"""

        self.__run_example("modem_waveforms_ocdm.py")

    def test_modem_waveforms_otfs(self) -> None:
        """Test example snippet for OTFS waveforms"""

        self.__run_example("modem_waveforms_otfs.py")

    def test_modem(self) -> None:
        """Test example snippet for modem"""

        self.__run_example("modem.py")

    def test_radar_evaluators_DetectionProbEvaluation(self) -> None:
        """Test example snippet for detection probability evaluation"""

        self.__run_example("radar_evaluators_DetectionProbEvaluator.py")

    def test_radar_evaluators_ReceiverOperatingCharacteristic(self) -> None:
        """Test example snippet for receiver oeperating characteristic evaluation"""

        self.__run_example("radar_evaluators_ReceiverOperatingCharacteristic.py")

    def test_radar_evaluators_RootMeanSquareError(self) -> None:
        """Test example snippet for root mean square error evaluation"""

        self.__run_example("radar_evaluators_RootMeanSquareError.py")

    def test_radar_fmcw_FMCW(self) -> None:
        """Test example snippet for FMCW waveforms"""

        self.__run_example("radar_fmcw_FMCW.py")

    def test_simulation_adc(self) -> None:
        """Test example snippet for ADC"""

        self.__run_example("simulation_adc.py")

    def test_simulation_antennas_SimulatedAntennaArray(self) -> None:
        """Test example snippet for simulated antenna array"""

        self.__run_example("simulation_antennas_SimulatedAntennaArray.py")

    def test_simulation_antennas(self) -> None:
        """Test example snippet for antennas"""

        self.__run_example("simulation_antennas.py")

    def test_simulation_amplifier(self) -> None:
        """Test example snippet for amplifier"""

        self.__run_example("simulation_amplifier.py")

    def test_simulation_animation(self) -> None:
        """Test example snippet for animation"""

        self.__run_example("simulation_animation.py")

    def test_simulation_isolation(self) -> None:
        """Test example snippet for isolation"""

        self.__run_example("simulation_isolation.py")

    def test_simulation_isolation_selective(self) -> None:
        """Test example snippet for selective isolation"""

        self.__run_example("simulation_isolation_selective.py")

    def test_simulation_isolation_specific(self) -> None:
        """Test example snippet for specific isolation"""

        self.__run_example("simulation_isolation_specific.py")

    def test_simulation_isolation_perfect(self) -> None:
        """Test example snippet for perfect isolation"""

        self.__run_example("simulation_isolation_perfect.py")

    def test_simulation_noise(self) -> None:
        """Test example snippet for noise"""

        self.__run_example("simulation_noise.py")

    def test_simulation_phase_noise(self) -> None:
        """Test example snippet for phase noise"""

        self.__run_example("simulation_phase_noise.py")

    def test_simulation_SimulatedDevice(self) -> None:
        """Test example snippet for simulated device"""

        self.__run_example("simulation_SimulatedDevice.py")

    def test_simulation_synchronization_RandomTrigger(self) -> None:
        """Test example snippet for random trigger"""

        self.__run_example("simulation_synchronization_RandomTrigger.py")

    def test_simulation_synchronization_SampleOffsetTrigger(self) -> None:
        """Test example snippet for sample offset trigger"""

        self.__run_example("simulation_synchronization_SampleOffsetTrigger.py")

    def test_simulation_synchronization_StaticTrigger(self) -> None:
        """Test example snippet for static trigger"""

        self.__run_example("simulation_synchronization_StaticTrigger.py")

    def test_simulation_synchronization_TimeOffsetTrigger(self) -> None:
        """Test example snippet for time offset trigger"""

        self.__run_example("simulation_synchronization_TimeOffsetTrigger.py")

    def test_simulation_synchronization(self) -> None:
        """Test example snippet for synchronization"""

        self.__run_example("simulation_synchronization.py")

    def test_simulation(self) -> None:
        """Test example snippet for simulation"""

        self.__run_example("simulation.py")
