# -*- coding: utf-8 -*-

from ipaddress import ip_address
from itertools import product
from os import mkdir
from os.path import join

import numpy as np

from hermespy.simulation import Simulation
from hermespy.modem import BitErrorEvaluator, DuplexModem, RootRaisedCosineWaveform, ChirpFSKWaveform, OFDMWaveform, GridResource, SymbolSection, GridElement
from hermespy.tools import db2lin
from hermespy.channel import Channel
from hermespy.core import ConsoleMode

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"

waveforms = [
    ('fsk', ChirpFSKWaveform(oversampling_factor=4, num_data_chirps=100, num_pilot_chirps=0, chirp_duration=4e-6, chirp_bandwidth=200e6)),
    ('sc', RootRaisedCosineWaveform(oversampling_factor=1, symbol_rate=100e6, num_data_symbols=100, num_preamble_symbols=0)),
    ('ofdm', OFDMWaveform(oversampling_factor=1, resources=[GridResource(1200, 0., [GridElement('DATA')])], structure=[SymbolSection(1, [0])])),
]

channel = Channel()
modulation_orders = [2, 4, 16, 64]

for modulation_order, (waveform_name, waveform) in product(modulation_orders, waveforms):

    simulation = Simulation(num_actors=12)
    device = simulation.scenario.new_device()

    modem = DuplexModem()
    modem.device = device
    modem.waveform = waveform
    modem.waveform.modulation_order = modulation_order
    simulation.scenario.set_channel(device, device, channel)

    simulation.new_dimension('noise_level', [db2lin(x) for x in np.arange(-10, 20, .5)])
    simulation.add_evaluator(BitErrorEvaluator(modem, modem))
    simulation.num_samples, simulation.min_num_samples = 100000, 100000
    simulation.plot_results = True
    simulation.results_dir = simulation.default_results_dir()
    _ = simulation.run()
