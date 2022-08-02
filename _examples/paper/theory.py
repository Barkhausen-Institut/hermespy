from itertools import product
from os import mkdir
from os.path import join

import numpy as np

from hermespy.simulation import Simulation
from hermespy.modem import BitErrorEvaluator, Modem, WaveformGeneratorPskQam, WaveformGeneratorChirpFsk, WaveformGeneratorOfdm, FrameResource, FrameSymbolSection, FrameElement
from hermespy.tools import db2lin
from hermespy.channel import Channel, MultipathFadingChannel
from hermespy.core import ConsoleMode


waveforms = [
    ('sc', WaveformGeneratorPskQam(oversampling_factor=1, num_data_symbols=100, num_preamble_symbols=0)),
    ('ofdm', WaveformGeneratorOfdm(oversampling_factor=1, resources=[FrameResource(1200, 0., [FrameElement('DATA')])], structure=[FrameSymbolSection(1, [0])])),
    ('fsk', WaveformGeneratorChirpFsk(oversampling_factor=1, num_data_chirps=100, num_pilot_chirps=0)),
]

channels = [
    ('awgn', Channel()),
#    ('rayleigh', MultipathFadingChannel([0.], [1.], [0.])),
]

modulation_orders = [2, 4, 16, 64]
directory_prefix = '/home/jan.adler/paper/validation/'

for modulation_order, (waveform_name, waveform), (channel_name, channel) in product(modulation_orders, waveforms, channels):

    simulation = Simulation()
    simulation.num_actors = 18
    device = simulation.scenario.new_device()

    modem = Modem()
    modem.device = device
    modem.waveform_generator = waveform
    modem.waveform_generator.modulation_order = modulation_order
    simulation.scenario.set_channel(device, device, channel)

    simulation.new_dimension('snr', [db2lin(x) for x in np.arange(-10, 20, .5)])
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
