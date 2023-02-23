from numpy import linspace

from hermespy.hardware_loop import HardwareLoop, UsrpSystem
from hermespy.modem import SimplexLink, RRCWaveform, SCCorrelationSynchronization, SCLeastSquaresChannelEstimation, SCZeroForcingChannelEqualization, BitErrorEvaluator


loop = HardwareLoop(UsrpSystem(), manual_triggering=False, plot_information=False)
loop.verbosity = "ALL"

# Configure a single carrier waveform modulated with 16-QAM root-raised cosine 
# shaped pulses at a rate of 61.44 MHz 
wave = RRCWaveform(symbol_rate=61.44e6, num_preamble_symbols=100,
                   num_data_symbols=1000, modulation_order=2)

# Configure the waveform to perform a correlation-based synchronization followed
# by a least-squares channel estimation and zero-forcing equalization
wave.synchronization = SCCorrelationSynchronization()
wave.channel_estimation = SCLeastSquaresChannelEstimation()
wave.channel_equalization = SCZeroForcingChannelEqualization()

# Configure device nodes, the base carrier frequency should be 1GHz
device_configuration = {
    'carrier_frequency': 3.7e9,
    'tx_gain': 30,
    'rx_gain': 10,
    'max_receive_delay': 2e-6,
    'calibration_delay':  5e-7,
}

transmitting_device = loop.new_device("192.168.189.131", **device_configuration)
receiving_device = loop.new_device("192.168.189.132", **device_configuration)

# @jan.adler if this line is commented out, the code breaks afterwards, because somehow
# the filter BW is not correct
# print(device.estimate_noise_power())

# Initialize a communicationlink between two dedicated devices
link = SimplexLink(transmitting_device, receiving_device, waveform=wave)

# Configure a bit error rate evaluation for each link
ber = BitErrorEvaluator(link, link)
loop.add_evaluator(ber)
    
# Configure  a parameter sweep over the TX gain (i.e. transmit power) of the device
gains = linspace(0, 40, 5, endpoint=True)
loop.new_dimension('tx_gain', gains, transmitting_device)

# Execute the pipeline
loop.num_drops = 3
loop.run()
