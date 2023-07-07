# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np

from hermespy.core import Transformation
from hermespy.simulation import RandomTrigger, SimulationScenario, StaticTrigger
from hermespy.channel import StreetCanyonLineOfSight
from hermespy.modem import BitErrorEvaluator, SimplexLink, RootRaisedCosineWaveform, RectangularWaveform, SingleCarrierIdealChannelEstimation, SingleCarrierZeroForcingChannelEqualization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestTriggerGroups(TestCase):
    """Test randomized trigger groups over multiple links and channel models"""

    def setUp(self) -> None:
        
        self.trigger_alpha_link = RandomTrigger()
        self.trigger_beta_link = RandomTrigger()

        self.carrier_frequency = 75e9
        self.scenario = SimulationScenario(seed=42)

        # Set up the base scenario devices
        self.alpha_transmitter = self.scenario.new_device(carrier_frequency=self.carrier_frequency, trigger_model=self.trigger_alpha_link, pose=Transformation.From_Translation(np.array([100, 0, 0])))
        self.alpha_receiver = self.scenario.new_device(carrier_frequency=self.carrier_frequency, trigger_model=self.trigger_alpha_link, pose=Transformation.From_Translation(np.array([0, 100, 0])))
        self.beta_transmitter = self.scenario.new_device(carrier_frequency=self.carrier_frequency, trigger_model=self.trigger_beta_link, pose=Transformation.From_Translation(np.array([-100, 0, 0])))
        self.beta_receiver = self.scenario.new_device(carrier_frequency=self.carrier_frequency, trigger_model=self.trigger_beta_link, pose=Transformation.From_Translation(np.array([0, -100, 0])))

        # Set up the linking channel models
        self.scenario.set_channel(self.alpha_transmitter, self.alpha_receiver, StreetCanyonLineOfSight())
        self.scenario.set_channel(self.beta_transmitter, self.beta_receiver, StreetCanyonLineOfSight())
        self.scenario.channel(self.alpha_transmitter, self.beta_receiver).gain = 0.
        self.scenario.channel(self.beta_transmitter, self.alpha_receiver).gain = 0.

        # Set up the operations
        self.alpha_link = SimplexLink(self.alpha_transmitter, self.alpha_receiver,
                                      waveform=RectangularWaveform(guard_interval=4e-7, modulation_order=4, num_preamble_symbols=0, num_data_symbols=100, symbol_rate=3e7, pilot_rate=1e7, channel_estimation=SingleCarrierIdealChannelEstimation(), channel_equalization=SingleCarrierZeroForcingChannelEqualization()))
        self.beta_link = SimplexLink(self.beta_transmitter, self.beta_receiver,
                                     waveform=RectangularWaveform(guard_interval=4e-7, modulation_order=4, num_preamble_symbols=0, num_data_symbols=100, symbol_rate=3e7, pilot_rate=1e7, channel_estimation=SingleCarrierIdealChannelEstimation(), channel_equalization=SingleCarrierZeroForcingChannelEqualization()))

        # Set up evaluators
        self.alpha_error = BitErrorEvaluator(self.alpha_link, self.alpha_link)
        self.beta_error = BitErrorEvaluator(self.beta_link, self.beta_link)

    def test_drop(self) -> None:
        """Test the drop generation and resulting error rates with randomized trigger groups"""

        for _ in range(10):

            drop = self.scenario.drop()

            # Make sure the shared devices have the same trigger realization
            self.assertIs(drop.device_transmissions[0].trigger_realization, drop.device_receptions[1].trigger_realization)
            self.assertIs(drop.device_transmissions[2].trigger_realization, drop.device_receptions[3].trigger_realization)

            self.assertGreaterEqual(0.01, self.alpha_error.evaluate().artifact().to_scalar())
            self.assertGreaterEqual(0.01, self.beta_error.evaluate().artifact().to_scalar())