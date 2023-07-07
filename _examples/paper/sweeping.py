# -*- coding: utf-8 -*-

from ipaddress import ip_address
from itertools import product
from os import mkdir
from os.path import join
from wsgiref.handlers import read_environ

import numpy as np
from hermespy.modem import SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization

from hermespy.simulation import Simulation
from hermespy.modem import BitErrorEvaluator, DuplexModem, RootRaisedCosineWaveform
from hermespy.tools import db2lin
from hermespy.channel import MultipathFading5GTDL
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
simulation = Simulation(console_mode=ConsoleMode.LINEAR, ray_address='auto') if cluster else Simulation(num_actors=10)

device = simulation.scenario.new_device()
channel = MultipathFading5GTDL(doppler_frequency=1000)
simulation.scenario.set_channel(device, device, channel)

modem = DuplexModem()
modem.device = device
sc = RootRaisedCosineWaveform(oversampling_factor=1, symbol_rate=100e6, num_data_symbols=1000, num_preamble_symbols=1, num_postamble_symbols=1)
sc.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
sc.channel_equalization = SingleCarrierZeroForcingChannelEqualization()
modem.waveform_generator = sc

simulation.results_dir = simulation.default_results_dir()
simulation.new_dimension('pilot_rate', [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512], sc)
simulation.new_dimension('doppler_frequency', np.linspace(0, 200e6, 21, endpoint=True), channel)
simulation.add_evaluator(BitErrorEvaluator(modem, modem))
simulation.num_samples, simulation.min_num_samples = 10000, 10000

simulation.run()
