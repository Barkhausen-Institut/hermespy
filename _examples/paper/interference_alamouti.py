import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock
from scipy.constants import speed_of_light

from hermespy.channel import IndoorFactoryLineOfSight, MultipathFading5GTDL
from hermespy.modem.waveform_generator_ofdm import FrameElement, FrameSymbolSection
from hermespy.modem import Modem, WaveformGeneratorOfdm, FrameResource, BitErrorEvaluator
from hermespy.modem.waveform_generator_psk_qam import PskQamLeastSquaresChannelEstimation, WaveformGeneratorPskQam
from hermespy.precoding.zero_forcing_equalizer import ZFTimeEqualizer
from hermespy.simulation import Simulation, SimulationScenario
from hermespy.simulation.simulation import SimulationRunner
from hermespy.core import ConsoleMode, UniformArray, IdealAntenna
from hermespy.radar import FMCW, Radar
from hermespy.tools import db2lin
from hermespy.precoding import SpaceTimeBlockCoding

# Initialize devices
carrier_frequency = 3.5e9
wavelength = speed_of_light / carrier_frequency
simulation = Simulation(console_mode=ConsoleMode.LINEAR, ray_address='auto')
tx_device = simulation.scenario.new_device(carrier_frequency=carrier_frequency, antennas=UniformArray(IdealAntenna(), .5 * wavelength, [2]))
rx_device = simulation.scenario.new_device(carrier_frequency=carrier_frequency, antennas=UniformArray(IdealAntenna(), .5 * wavelength, [2]))

in_device = simulation.scenario.new_device(carrier_frequency=(carrier_frequency+1e6))
tx_device.position = np.array([0., 0., 0.])
rx_device.position = np.array([40., 30., 0.])

# Configure waveforms and device operators
ofdm_resources = [
    FrameResource(200, 0.078125, [FrameElement('REFERENCE', 1), FrameElement('DATA', 5)]),
    FrameResource(1200, 0.0703125, [FrameElement('DATA')]),
    FrameResource(100, 0.0703125, [FrameElement('DATA', 3), FrameElement('REFERENCE', 1), FrameElement('DATA', 5), FrameElement('REFERENCE', 1), FrameElement('DATA', 2)]),
]
ofdm_transmit_tructure = [
    FrameSymbolSection(16, [0, 1, 1, 1, 2, 1, 1]),
]
ofdm_receive_tructure = [
    FrameSymbolSection(16, [0, 1, 1, 1, 2, 1, 1]),
]

transmit_operator = Modem()
transmit_operator.waveform_generator = WaveformGeneratorOfdm(modulation_order=256, subcarrier_spacing=15e3, dc_suppression=False, num_subcarriers=2048, resources=ofdm_resources, structure=ofdm_transmit_tructure, oversampling_factor=1)
transmit_operator.precoding[0] = SpaceTimeBlockCoding()
transmit_operator.device = tx_device
receive_operator = Modem()
receive_operator.waveform_generator = WaveformGeneratorOfdm(modulation_order=256, subcarrier_spacing=15e3, dc_suppression=False, num_subcarriers=2048, resources=ofdm_resources, structure=ofdm_receive_tructure, oversampling_factor=1)
receive_operator.device = rx_device
receive_operator.precoding[0] = SpaceTimeBlockCoding()
transmit_operator.reference_transmitter = receive_operator
receive_operator.reference_transmitter = transmit_operator
interfering_opeartor = Radar()
interfering_opeartor.waveform = FMCW(sampling_rate=transmit_operator.waveform_generator.sampling_rate,
                                     bandwidth=transmit_operator.waveform_generator.bandwidth,
                                     max_range=transmit_operator.waveform_generator.samples_in_frame * speed_of_light / (2 * transmit_operator.waveform_generator.bandwidth))
interfering_opeartor.waveform.num_chirps = 1
interfering_opeartor.device = in_device

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
simulation.new_dimension('snr', [db2lin(x) for x in np.arange(-10, 20, .5)])
simulation.num_samples, simulation.min_num_samples = 10000, 10000
simulation.plot_results = True
simulation.scenario.set_seed(42)
simulation.results_dir = '/home/jan.adler/paper/alamouti/'

simulation.run()