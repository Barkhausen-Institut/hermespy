# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np

from hermespy.core import Transformation
from hermespy.simulation import RandomTrigger, Simulation, SimulationScenario
from hermespy.channel import O2IState, UrbanMacrocells
from hermespy.modem import BitErrorEvaluator, SimplexLink, OFDMWaveform, GridElement, GridResource, ElementType, SymbolSection, PilotSection, OFDMCorrelationSynchronization, OrthogonalLeastSquaresChannelEstimation, OrthogonalZeroForcingChannelEqualization
from unit_tests.utils import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestTriggerGroups(TestCase):
    """Test randomized trigger groups over multiple links and channel models"""

    def __ofdm_waveform(self) -> OFDMWaveform:
        """Configure an OFDM wafeform with default parameters.

        Returns: The configured waveform.
        """

        # Mock 5G numerology #1:
        # 120khz subcarrier spacing, 120 subcarriers, 2us guard interval, 1ms subframe duration

        prefix_ratio = 0.0684
        num_subcarriers = 128
        grid_resources = [
            GridResource(num_subcarriers // 5, prefix_ratio=prefix_ratio, elements=[
                GridElement(ElementType.REFERENCE, 1),
                GridElement(ElementType.DATA, 4)
            ]),
            GridResource(num_subcarriers // 5, prefix_ratio=prefix_ratio, elements=[
                GridElement(ElementType.DATA, 2),
                GridElement(ElementType.REFERENCE, 1),
                GridElement(ElementType.DATA, 2),
            ]),
        ]
        grid_structure = [
            SymbolSection(
                num_subcarriers // 2,
                [0, 1],
                5,
            ),
        ]

        waveform = OFDMWaveform(subcarrier_spacing=1e6 / num_subcarriers, num_subcarriers=num_subcarriers, dc_suppression=True, grid_resources=grid_resources, grid_structure=grid_structure)
        waveform.oversampling_factor = 2
        waveform.pilot_section = PilotSection()
        waveform.synchronization = OFDMCorrelationSynchronization()
        waveform.channel_estimation = OrthogonalLeastSquaresChannelEstimation()
        waveform.channel_equalization = OrthogonalZeroForcingChannelEqualization()
        #waveform.pilot_symbol_sequence = CustomPilotSymbolSequence(np.arange(1, num_subcarriers * 60))

        return waveform

    def setUp(self) -> None:
        self.trigger_alpha_link = RandomTrigger()
        self.trigger_beta_link = RandomTrigger()

        self.carrier_frequency = 1e8
        self.scenario = SimulationScenario(seed=42)

        # Set up the base scenario devices
        self.alpha_transmitter = self.scenario.new_device(
            carrier_frequency=self.carrier_frequency,
            trigger_model=self.trigger_alpha_link,
            pose=Transformation.From_Translation(np.array([100, 0, 0]))
        )
        self.alpha_receiver = self.scenario.new_device(
            carrier_frequency=self.carrier_frequency,
            trigger_model=self.trigger_alpha_link,
            pose=Transformation.From_Translation(np.array([0, 100, 0])),
        )
        self.beta_transmitter = self.scenario.new_device(
            carrier_frequency=self.carrier_frequency,
            trigger_model=self.trigger_beta_link,
            pose=Transformation.From_Translation(np.array([-100, 0, 0])),
        )
        self.beta_receiver = self.scenario.new_device(
            carrier_frequency=self.carrier_frequency,
            trigger_model=self.trigger_beta_link,
            pose=Transformation.From_Translation(np.array([0, -100, 0])),
        )

        # Set up the linking channel models
        channel = UrbanMacrocells(expected_state=O2IState.LOS, seed=42)
        self.scenario.set_channel(self.alpha_receiver, self.alpha_transmitter, channel)
        self.scenario.set_channel(self.beta_receiver, self.beta_transmitter, channel)
        self.scenario.channel(self.alpha_transmitter, self.beta_receiver).gain = 0.0
        self.scenario.channel(self.beta_transmitter, self.alpha_receiver).gain = 0.0

        # Set up the operations
        self.alpha_link = SimplexLink(
            self.alpha_transmitter, self.alpha_receiver,
            waveform=self.__ofdm_waveform(),
        )
        self.beta_link = SimplexLink(
            self.beta_transmitter, self.beta_receiver,
            waveform=self.__ofdm_waveform(),
        )

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

            alpha_ber = self.alpha_error.evaluate().artifact().to_scalar()
            beta_ber = self.beta_error.evaluate().artifact().to_scalar()
            self.assertGreaterEqual(0.1, alpha_ber)
            self.assertGreaterEqual(0.1, beta_ber)

    def test_pseudo_randomness(self) -> None:
        """Setting the scenario's random seed should yield reproducable trigger realizations"""

        num_drops = 5

        self.scenario.seed = 42
        initial_realizations = []
        for _ in range(num_drops):
            drop = self.scenario.drop()
            initial_realizations.append(drop.device_transmissions[0].trigger_realization)

        self.scenario.seed = 42
        replayed_realizations = []
        for _ in range(num_drops):
            drop = self.scenario.drop()
            replayed_realizations.append(drop.device_transmissions[0].trigger_realization)   

        for initial_realization, replayed_realization in zip(initial_realizations, replayed_realizations):
            self.assertEqual(initial_realization.num_offset_samples, replayed_realization.num_offset_samples)

    def test_simulation(self) -> None:
        """Trigger groups should be correctly applied during simulation runtime"""

        with SimulationTestContext():

            simulation = Simulation(self.scenario)
            simulation.num_drops = 10
            simulation.add_evaluator(self.alpha_error)
            simulation.add_evaluator(self.beta_error)

            result = simulation.run()

            self.assertGreaterEqual(0.01, result.evaluation_results[0].to_array().flat[0])
            self.assertGreaterEqual(0.01, result.evaluation_results[1].to_array().flat[0])
