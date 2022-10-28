# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from metakernel import Magic

from hermespy.hardware_loop.uhd import UsrpSystem

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestUsrpSystem(TestCase):
    """Test the USRP system binding to Hermes."""
    
    def setUp(self) -> None:
        
        self.device_patch = patch('hermespy.hardware_loop.uhd.system.UsrpDevice')
        self.device_mock: MagicMock = self.device_patch.start()
        
        self.system = UsrpSystem()

    def tearDown(self) -> None:
        
        self.device_patch.stop()

    def test_new_device(self) -> None:
        """Test the creation of a new device"""
        
        ip = '123.456.789.012'
        port = 1234

        new_device = self.system.new_device(ip=ip, port=port)
        
        self.assertTrue(self.system.device_registered(new_device))
        
    def test_add_device(self) -> None:
        """Test the registration of an existing device"""
        
        added_device = Mock()
        self.system.add_device(added_device)
        
        self.assertTrue(self.system.device_registered(added_device))
        
    @patch('usrp_client.system.System.execute')
    def test_trigger(self, execute: MagicMock) -> None:
        """Test the system triggering"""
        
        self.system._trigger()
        execute.assert_called_once()
        