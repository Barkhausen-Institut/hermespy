from os import path
from unittest import TestCase
from unittest.mock import patch, Mock
from tempfile import TemporaryDirectory

import ray
import numpy as np

from hermespy.simulation import Simulation
from hermespy.fec import LDPCCoding
from hermespy.modem import DuplexModem, RootRaisedCosineWaveform
from hermespy.tools import db2lin

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestMonteCarloFEC(TestCase):
    
    def setUp(self) -> None:
        
        self.simulation = Simulation()
        self.simulation.new_dimension('snr', [db2lin(x) for x in np.arange(-10, 10, .5)])
        self.simulation.num_samples = 2
        device = self.simulation.scenario.new_device()
        self.modem = DuplexModem()
        self.modem.device = device
        self.modem.waveform_generator = RootRaisedCosineWaveform(oversampling_factor=1, symbol_rate=100e6, num_data_symbols=200, modulation_order=64, num_preamble_symbols=0)
        
    @classmethod
    def setUpClass(cls) -> None:

        ray.init(local_mode=True, num_cpus=2, ignore_reinit_error=True)
            
    @classmethod
    def tearDownClass(cls) -> None:

        # Shut down ray 
        ray.shutdown()
        
    def __run_simulation(self) -> None:
        """Run a simulation and test for proper execution."""

        with patch('sys.stdout'), patch('matplotlib.pyplot.figure'):
            
            self.simulation.run()
    
    def _test_ldpc(self) -> None:
        
        ldpc_matrix = path.join(path.dirname(path.realpath(__file__)), '..', '..', 'submodules', 'affect', 'conf', 'dec', 'LDPC', 'CCSDS_64_128.alist')
        
        with TemporaryDirectory() as g_dir:
            
            g_path = path.join(g_dir, 'g_save.alist')
            coding = LDPCCoding(100, ldpc_matrix, "", False, 10)
            self.modem.encoder_manager.add_encoder(coding)
            
            self.__run_simulation()
