# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import List

from numpy import linspace

from hermespy.core import Device
from hermespy.simulation import Simulation
from hermespy.modem import TransmittingModem, ReceivingModem, RRCWaveform, SCCorrelationSynchronization, SCLeastSquaresChannelEstimation, SCZeroForcingChannelEqualization, BitErrorEvaluator

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


pipe = Simulation()

# Configure a single carrier waveform modulated with 16-QAM root-raised cosine 
# shaped pulses at a rate of 61.44e6 MHz 
wave = RRCWaveform(symbol_rate=61.44e6, num_preamble_symbols=16,
                   num_data_symbols=100, modulation_order=16)
                   
# Configure the waveform to perform a correlation-based synchronization followed
# by a least-squares channel estimation and zero-forcing equalization
wave.synchronization = SCCorrelationSynchronization()
wave.channel_estimation = SCLeastSquaresChannelEstimation()
wave.channel_equalization = SCZeroForcingChannelEqualization()

# Configure device nodes, the base carrier frequency should be 700 MHz
cf = 7e8
devs: List[Device] = []

# Initialize virtual devices within the simulation
for _ in range(4):

    device = pipe.new_device(carrier_frequency=cf)
    devs.append(device)

# Set up two interfering simplex links between a pair of devices, respectively
for d in range(2):
    
    # Initialize a new transmitting modem
    tx_operator = TransmittingModem()
    tx_operator.waveform = deepcopy(wave)
    
    # Initialize a new receiving modem
    rx_operator = ReceivingModem()
    rx_operator.waveform = deepcopy(wave)
    
    # Configure the devices by assigning them modem operators
    devs[d].transmitters.add(tx_operator)
    devs[d+2].receivers.add(rx_operator)

    # Configure a bit error rate evaluation for each link
    ber = BitErrorEvaluator(tx_operator, rx_operator)
    pipe.add_evaluator(ber)
    
# Configure  a parameter sweep over the two links' carrier frequency distance
dist = linspace(wave.bandwidth, 0., 101, endpoint=True) + cf
pipe.new_dimension('carrier_frequency', dist, devs[1], devs[3])

# Execute the pipeline
pipe.num_drops = 1000
pipe.results_dir = 'D:\\hermes_paper\\multinode\\simulation'
pipe.plot_results = True
pipe.run()
