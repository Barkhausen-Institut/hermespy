# -*- coding: utf-8 -*-

import numpy as np
from scipy.constants import pi

from hermespy.simulation import Simulation, RappPowerAmplifier
from hermespy.channel import MultipathFading5GTDL
from hermespy.modem import BitErrorEvaluator, DuplexModem, RaisedCosineWaveform, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization
from hermespy.tools import db2lin
from hermespy.core import ConsoleMode

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


cluster = False

simulation = Simulation(console_mode=ConsoleMode.LINEAR, ray_address='auto') if cluster else Simulation()
device = simulation.scenario.new_device(carrier_frequency=3.7e9)
simulation.scenario.set_channel(device, device, MultipathFading5GTDL())

modem = DuplexModem()
modem.device = device
modem.waveform = RaisedCosineWaveform(oversampling_factor=4, num_preamble_symbols=16, num_data_symbols=100, symbol_rate=100e6, modulation_order=16)
modem.waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
modem.waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

simulation.new_dimension('noise_level', [db2lin(x) for x in np.arange(-10, 20, .5)])
simulation.add_evaluator(BitErrorEvaluator(modem, modem))
simulation.num_samples, simulation.min_num_samples = 100000, 100000
simulation.plot_results = True

# Ideal hardware
simulation.results_dir = '/home/jan.adler/paper/hardware_effects/ideal'  if cluster else 'D:\\hermes_paper\\hardware_effects\\ideal'
simulation.run()

# Power amplifier imperfections
device.rf_chain.power_amplifier = RappPowerAmplifier(saturation_amplitude=1.)
simulation.results_dir = '/home/jan.adler/paper/hardware_effects/pa' if cluster else 'D:\\hermes_paper\\hardware_effects\\pa'
simulation.run()
device.rf_chain.power_amplifier = None

# I/Q imbalance imperfections
device.rf_chain.amplitude_imbalance = .05
device.rf_chain.phase_offset = pi / 180
simulation.results_dir = '/home/jan.adler/paper/hardware_effects/iq' if cluster else 'D:\\hermes_paper\\hardware_effects\\iq'
simulation.run()
device.rf_chain.amplitude_imbalance = 0.
device.rf_chain.phase_offset = 0.

# ADC quantization imperfections
device.adc.num_quantization_bits = 16
simulation.results_dir = '/home/jan.adler/paper/hardware_effects/adc' if cluster else 'D:\\hermes_paper\\hardware_effects\\adc'
simulation.run()
device.adc.num_quantization_bits = np.inf

# Full imperfect assumptions
device.rf_chain.power_amplifier = RappPowerAmplifier(saturation_amplitude=1.)
device.rf_chain.amplitude_imbalance = .05
device.rf_chain.phase_offset = pi / 180
device.adc.num_quantization_bits = 16
simulation.results_dir = '/home/jan.adler/paper/hardware_effects/all' if cluster else 'D:\\hermes_paper\\hardware_effects\\all'
simulation.run()
