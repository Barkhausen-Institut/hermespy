# -*- coding: utf-8 -*-

from hermespy.hardware_loop import HardwareLoop, UsrpSystem, UsrpDevice
from hermespy.modem import SimplexLink, RRCWaveform, SCCorrelationSynchronization, SCLeastSquaresChannelEstimation, SCZeroForcingChannelEqualization, BitErrorEvaluator
from hermespy.core import Verbosity

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


loop = HardwareLoop[UsrpSystem, UsrpDevice](UsrpSystem(), manual_triggering=False, plot_information=True)
loop.verbosity = Verbosity.INFO

# Configure a single carrier waveform modulated with 16-QAM root-raised cosine pulses
wave = RRCWaveform(
    num_preamble_symbols=100,
    num_data_symbols=1000,
    modulation_order=16,
)

# Configure the waveform to perform a correlation-based synchronization followed
# by a least-squares channel estimation and zero-forcing equalization
wave.synchronization = SCCorrelationSynchronization()
wave.channel_estimation = SCLeastSquaresChannelEstimation()
wave.channel_equalization = SCZeroForcingChannelEqualization()

# Configure device nodes, the base carrier frequency should be 1GHz
device_configuration = {
    'carrier_frequency': 3.7e9,
    'tx_gain': 10,
    'rx_gain': 40,
    'max_receive_delay': 2e-6,
#    'calibration_delay': 5e-7,
}

transmitting_device = loop.new_device("192.168.189.131", **device_configuration)
receiving_device = loop.new_device("192.168.189.132", **device_configuration)

# Initialize a communicationlink between two dedicated devices
link = SimplexLink()
link.waveform = wave
transmitting_device.transmitters.add(link)
receiving_device.receivers.add(link)

# Configure a bit error rate evaluation for each link
ber = BitErrorEvaluator(link, link)
loop.add_evaluator(ber)

# Configure  a parameter sweep over the TX gain (i.e. transmit power) of the device
#gains = linspace(0, 40, 5, endpoint=True)
#loop.new_dimension('tx_gain', gains, transmitting_device)

# Execute the pipeline
loop.num_drops = 100
loop.run()
