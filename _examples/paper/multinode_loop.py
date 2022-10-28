from copy import deepcopy
from ipaddress import IPv4Address
from typing import List

from numpy import linspace

from hermespy.core import Device
from hermespy.hardware_loop import HardwareLoop, UsrpSystem
from hermespy.modem import TransmittingModem, ReceivingModem, RRCWaveform, SCCorrelationSynchronization, SCLeastSquaresChannelEstimation, SCZeroForcingChannelEqualization, BitErrorEvaluator

pipe = HardwareLoop(UsrpSystem(), manual_triggering=False)

# Configure a single carrier waveform modulated with 16-QAM root-raised cosine 
# shaped pulses at a rate of 61.44 MHz 
wave = RRCWaveform(symbol_rate=61.44e6, num_preamble_symbols=16,
                   num_data_symbols=100, modulation_order=16)
                   
# Configure the waveform to perform a correlation-based synchronization followed
# by a least-squares channel estimation and zero-forcing equalization
wave.synchronization = SCCorrelationSynchronization()
wave.channel_estimation = SCLeastSquaresChannelEstimation()
wave.channel_equalization = SCZeroForcingChannelEqualization()

# Configure device nodes, the base carrier frequency should be 1GHz
cf = 3.7e9
devs: List[Device] = []

# Initialize physical devices at remote IPs to be controlled by Hermes
for d in range(4):

    ip = str(IPv4Address("192.168.189.131") + d)
    device = pipe.new_device(ip=ip, carrier_frequency=cf, tx_gain=30., rx_gain=50.,
                             max_receive_delay=2e-6, calibration_delay=.5e-6)
    devs.append(device)

# Set up two interfering simplex links between a pair of devices, respectively
for d in range(2):
    
    # Initialize a new transmitting modem
    tx_operator = TransmittingModem()
    tx_operator.waveform_generator = deepcopy(wave)

    # Hack for the mismatching sampling rates
    if d == 0:
        tx_operator.waveform_generator.oversampling_factor = 8
    
    # Initialize a new receiving modem
    rx_operator = ReceivingModem()
    rx_operator.waveform_generator = deepcopy(wave)
    
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
pipe.results_dir = 'D:\\hermes_paper\\multinode\\hardware_loop'
pipe.plot_results = True
pipe.run()
