# -*- coding: utf-8 -*-

import logging
import os.path as path
from unittest import TestCase

import ray as ray

from ..helpers import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
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

    def test_channel_cdl_indoor_factory_los(self) -> None:
        """Test example snippet for indoor channel factory with LOS"""

        self.__run_example("channel_cdl_indoor_factory_los.py")

    def test_channel_cdl_indoor_factory_nlos(self) -> None:
        """Test example snippet for indoor channel factory with NLOS"""

        self.__run_example("channel_cdl_indoor_factory_nlos.py")

    def test_channel_cdl_indoor_office_los(self) -> None:
        """Test example snippet for indoor office channel with LOS"""

        self.__run_example("channel_cdl_indoor_office_los.py")

    def test_channel_cdl_indoor_office_nlos(self) -> None:
        """Test example snippet for indoor office channel with NLOS"""

        self.__run_example("channel_cdl_indoor_office_nlos.py")

    def test_channel_cdl_rural_macrocells_los(self) -> None:
        """Test example snippet for rural macrocells channel with LOS"""

        self.__run_example("channel_cdl_rural_macrocells_los.py")

    def test_channel_cdl_rural_macrocells_nlos(self) -> None:
        """Test example snippet for rural macrocells channel with NLOS"""

        self.__run_example("channel_cdl_rural_macrocells_nlos.py")

    def test_channel_cdl_street_canyon_los(self) -> None:
        """Test example snippet for street canyon channel with LOS"""

        self.__run_example("channel_cdl_street_canyon_los.py")

    def test_channel_cdl_street_canyon_nlos(self) -> None:
        """Test example snippet for street canyon channel with NLOS"""

        self.__run_example("channel_cdl_street_canyon_nlos.py")

    def test_channel_cdl_street_canyon_o2i(self) -> None:
        """Test example snippet for street canyon channel with O2I"""

        self.__run_example("channel_cdl_street_canyon_o2i.py")

    def test_channel_cdl_urban_macrocells_los(self) -> None:
        """Test example snippet for urban macrocells channel with LOS"""

        self.__run_example("channel_cdl_urban_macrocells_los.py")

    def test_channel_cdl_urban_macrocells_nlos(self) -> None:
        """Test example snippet for urban macrocells channel with NLOS"""

        self.__run_example("channel_cdl_urban_macrocells_nlos.py")

    def test_channel_cdl_urban_macrocells_o2i(self) -> None:
        """Test example snippet for urban macrocells channel with O2I"""

        self.__run_example("channel_cdl_urban_macrocells_o2i.py")

    def test_channel_MultipathFading5GTDL(self) -> None:
        """Test example snippet for 5G TDL multipath fading channel"""

        self.__run_example("channel_MultipathFading5GTDL.py")

    def test_channel_MultipathFadingChannel(self) -> None:
        """Test example snippet for multipath fading channel"""

        self.__run_example("channel_MultipathFadingChannel.py")

    def test_channel_MultipathFadingCost259(self) -> None:
        """Test example snippet for COST 259 multipath fading channel"""

        self.__run_example("channel_MultipathFadingCost259.py")

    def test_channel_MultipathFadingExponential(self) -> None:
        """Test example snippet for exponential multipath fading channel"""

        self.__run_example("channel_MultipathFadingExponential.py")

    def test_channel_MultiTargetRadarChannel(self) -> None:
        """Test example snippet for multi-target radar channel"""

        self.__run_example("channel_MultiTargetRadarChannel.py")

    def test_channel_RandomDelayChannel(self) -> None:
        """Test example snippet for random delay channel"""

        self.__run_example("channel_RandomDelayChannel.py")

    def test_channel_SingleTargetRadarChannel(self) -> None:
        """Test example snippet for single-target radar channel"""

        self.__run_example("channel_SingleTargetRadarChannel.py")

    def test_channel_SpatialDelayChannel(self) -> None:
        """Test example snippet for spatial delay channel"""

        self.__run_example("channel_SpatialDelayChannel.py")

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

    def test_modem_precoding_sm(self) -> None:
        """Test example snippet for spatial multiplexing precoding"""

        self.__run_example('modem_precoding_sm.py')

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
