# -*- coding: utf-8 -*-

import numpy as np
from scipy.constants import speed_of_light

from hermespy.channel import IndoorFactoryLineOfSight
from hermespy.modem.waveform_ofdm import FrameElement, FrameSymbolSection
from hermespy.modem import TransmittingModem, ReceivingModem, OFDMWaveform, GridResource, BitErrorEvaluator
from hermespy.simulation import Simulation
from hermespy.core import UniformArray, IdealAntenna
from hermespy.radar import FMCW, Radar
from hermespy.tools import db2lin
from hermespy.precoding import SpaceTimeBlockCoding

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


# Initialize devices
carrier_frequency = 3.5e9
wavelength = speed_of_light / carrier_frequency
simulation = Simulation()
tx_device = simulation.scenario.new_device(carrier_frequency=carrier_frequency, antennas=UniformArray(IdealAntenna(), .5 * wavelength, [2]))
rx_device = simulation.scenario.new_device(carrier_frequency=carrier_frequency, antennas=UniformArray(IdealAntenna(), .5 * wavelength, [2]))

in_device = simulation.scenario.new_device(carrier_frequency=(carrier_frequency+1e6))
tx_device.position = np.array([0., 0., 0.])
rx_device.position = np.array([40., 30., 0.])

# Configure waveforms and device operators
ofdm_resources = [
    GridResource(200, 0.078125, [FrameElement('REFERENCE', 1), FrameElement('DATA', 5)]),
    GridResource(1200, 0.0703125, [FrameElement('DATA')]),
    GridResource(100, 0.0703125, [FrameElement('DATA', 3), FrameElement('REFERENCE', 1), FrameElement('DATA', 5), FrameElement('REFERENCE', 1), FrameElement('DATA', 2)]),
]
ofdm_transmit_tructure = [
    FrameSymbolSection(16, [0, 1, 1, 1, 2, 1, 1]),
]
ofdm_receive_tructure = [
    FrameSymbolSection(16, [0, 1, 1, 1, 2, 1, 1]),
]

transmit_operator = TransmittingModem()
transmit_operator.waveform = OFDMWaveform(modulation_order=256, subcarrier_spacing=15e3, dc_suppression=False, num_subcarriers=2048, resources=ofdm_resources, structure=ofdm_transmit_tructure, oversampling_factor=1)
transmit_operator.precoding[0] = SpaceTimeBlockCoding()
tx_device.transmitters.add(transmit_operator)

receive_operator = ReceivingModem()
receive_operator.waveform = OFDMWaveform(modulation_order=256, subcarrier_spacing=15e3, dc_suppression=False, num_subcarriers=2048, resources=ofdm_resources, structure=ofdm_receive_tructure, oversampling_factor=1)
receive_operator.precoding[0] = SpaceTimeBlockCoding()
rx_device.receivers.add(receive_operator)

interfering_operator = Radar()
interfering_operator.waveform = FMCW(sampling_rate=transmit_operator.waveform.sampling_rate,
                                     bandwidth=transmit_operator.waveform.bandwidth,
                                     chirp_duration=transmit_operator.waveform.samples_in_frame * speed_of_light)
interfering_operator.waveform.num_chirps = 1
in_device.transmitters.add(interfering_operator)

# Configure channels
simulation.scenario.set_channel(rx_device, tx_device, IndoorFactoryLineOfSight(48000, 80000))
simulation.scenario.channel(tx_device, tx_device).gain = 0.
simulation.scenario.channel(rx_device, rx_device).gain = 0.
simulation.scenario.channel(in_device, in_device).gain = 0.
simulation.scenario.channel(tx_device, in_device).gain = 0.
simulation.scenario.channel(rx_device, in_device).gain = 1.

# Configure simulation
evaluator = BitErrorEvaluator(transmit_operator, receive_operator)
evaluator.tolerance = .0
evaluator.confidence = 1
simulation.add_evaluator(evaluator)
simulation.new_dimension('noise_level', [db2lin(x) for x in np.arange(-10, 20, .5)])
simulation.num_samples, simulation.min_num_samples = 10000, 10000
simulation.plot_results = True
simulation.scenario.set_seed(42)
simulation.results_dir = simulation.default_results_dir()

simulation.run()
