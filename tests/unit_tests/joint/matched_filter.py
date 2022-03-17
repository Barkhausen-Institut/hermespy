from unittest import TestCase, Mock

from hermespy.joint import MatchedFilterJoint

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "Jan Adler"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestMatchedFilterJoint(TestCase):
    """Matched filter joint testing."""
    
    def setUp(self) -> None:
        
        self.device = Mock()
        self.joint = MatchedFilterJoint()
        self.joint.device = self.device
        
    def test_transmit_receive(self) -> None:
        
        self.joint.transmit()
        self.joint.receive()