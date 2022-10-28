from copy import deepcopy

import numpy as np

from hermespy.fec.aff3ct import TurboCoding
from hermespy.tools import db2lin
from hermespy.simulation import Simulation

from hermespy.modem import TransmittingModem, ReceivingModem, OFDMWaveform, FrameResource, FrameElement, FrameSymbolSection, BitErrorEvaluator, BlockErrorEvaluator, FrameErrorEvaluator, ThroughputEvaluator


simulation = Simulation()

tx_device = simulation.new_device()
rx_device = simulation.new_device()

waveform = OFDMWaveform(oversampling_factor=1, modulation_order=64, resources=[FrameResource(1200, 0., [FrameElement('DATA')])], structure=[FrameSymbolSection(1, [0])])

tx_modem = TransmittingModem()
rx_modem = ReceivingModem()

tx_modem.waveform_generator = deepcopy(waveform) 
rx_modem.waveform_generator = deepcopy(waveform)
tx_modem.encoder_manager.add_encoder(TurboCoding(40, 13, 15, 10))
rx_modem.encoder_manager.add_encoder(TurboCoding(40, 13, 15, 10))


tx_device.transmitters.add(tx_modem)
rx_device.receivers.add(rx_modem)

simulation.new_dimension('snr', [db2lin(x) for x in np.arange(-10, 20, .5)])
simulation.add_evaluator(BitErrorEvaluator(tx_modem, rx_modem))
simulation.add_evaluator(BlockErrorEvaluator(tx_modem, rx_modem))
simulation.add_evaluator(FrameErrorEvaluator(tx_modem, rx_modem))
simulation.add_evaluator(ThroughputEvaluator(tx_modem, rx_modem))
simulation.num_samples, simulation.min_num_samples = 1000, 1000
simulation.plot_results = True

simulation.run()