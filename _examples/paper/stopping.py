# -*- coding: utf-8 -*-

from ipaddress import ip_address
from itertools import product
from os import mkdir
from os.path import join
from time import sleep

import numpy as np
from hermespy.modem.waveform_generator_ofdm import OFDMIdealChannelEstimation, OFDMLeastSquaresChannelEstimation, OFDMZeroForcingChannelEqualization

from hermespy.simulation import Simulation
from hermespy.modem import BitErrorEvaluator, DuplexModem, OFDMWaveform, FrameResource, FrameSymbolSection, FrameElement
from hermespy.tools import db2lin
from hermespy.channel import Channel
from hermespy.core import ConsoleMode

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


ofdm = OFDMWaveform(oversampling_factor=1, resources=[FrameResource(1200, 0., [FrameElement('DATA')])], structure=[FrameSymbolSection(1, [0])], modulation_order=64)
ofdm.channel_estimation = OFDMIdealChannelEstimation()
ofdm.channel_equalization = OFDMZeroForcingChannelEqualization()

channel = Channel()


confidences = np.linspace(.8, 1, 5)
tolerances = [1e-1, 1e-2, 1e-3, 1e-4]

for (c, confidence), (t, tolerance) in product(enumerate(confidences), enumerate(tolerances)):

    sleep(10)
    
    simulation = Simulation(console_mode=ConsoleMode.LINEAR, ray_address='auto', num_actors=800, seed=42) if cluster else Simulation()
    device = simulation.scenario.new_device()

    modem = DuplexModem()
    modem.device = device
    modem.waveform_generator = ofdm
    simulation.scenario.set_channel(device, device, channel)
    evaluator = BitErrorEvaluator(modem, modem)
    evaluator.confidence = confidence
    evaluator.tolerance = tolerance

    simulation.new_dimension('snr', [db2lin(x) for x in np.arange(-10, 20, .5)])
    simulation.add_evaluator(evaluator)
    simulation.num_samples = 100000
    simulation.min_num_samples = 100
    simulation.plot_results = True
    simulation.results_dir = simulation.default_results_dir()
    _ = simulation.run()

