# -*- coding: utf-8 -*-

from itertools import product
from os import mkdir
from os.path import join

import numpy as np

from hermespy.simulation import Simulation
from hermespy.modem import BitErrorEvaluator, DuplexModem, RootRaisedCosineWaveform, ChirpFSKWaveform, OFDMWaveform, GridResource, SymbolSection, GridElement
from hermespy.tools import db2lin
from hermespy.channel import Channel, MultipathFadingChannel
from hermespy.core import ConsoleMode

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


waveforms = [
    ('sc', RootRaisedCosineWaveform(oversampling_factor=1, num_data_symbols=100, num_preamble_symbols=0)),
    ('ofdm', ChirpFSKWaveform(oversampling_factor=1, resources=[GridResource(1200, 0., [GridElement('DATA')])], structure=[SymbolSection(1, [0])])),
    ('fsk', OFDMWaveform(oversampling_factor=1, num_data_chirps=100, num_pilot_chirps=0)),
]

channels = [
    ('awgn', Channel()),
    ('rayleigh', MultipathFadingChannel([0.], [1.], [0.])),
]

modulation_orders = [2, 4, 16, 64]
directory_prefix = Simulation.default_results_dir()

for modulation_order, (waveform_name, waveform), (channel_name, channel) in product(modulation_orders, waveforms, channels):

    simulation = Simulation()
    simulation.num_actors = 18
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
    
    result_dir = join(directory_prefix, channel_name, waveform_name, f'mod_{modulation_order}')
    try:
        mkdir(result_dir)
    except FileExistsError:
        ...
    
    simulation.results_dir = result_dir

    _ = simulation.run()
