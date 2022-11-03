# -*- coding: utf-8 -*-

from copy import deepcopy
from os import path

import numpy as np
from scipy.io import savemat

from hermespy.core import SNRType
from hermespy.channel import RadarChannel
from hermespy.jcas import MatchedFilterJcas
from hermespy.modem import OFDMWaveform, SchmidlCoxPilotSection
from hermespy.radar import Radar, FMCW, ReceiverOperatingCharacteristic
from hermespy.simulation import Simulation
from hermespy.tools import db2lin

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


bandwidth= 1e9

# Create a new simulation
simulation = Simulation()
simulation.num_samples = 10000
simulation.results_dir = simulation.default_results_dir()
simulation.plot_results = True
simulation.scenario.snr_type = SNRType.PN0
simulation.new_dimension('snr', [db2lin(snr) for snr in [-20, -10, 0]])

# Consider a device without interference in between operators
device_h1 = simulation.new_device(carrier_frequency=70e9)
device_h1.operator_separation = True
device_h0 = simulation.new_device(carrier_frequency=70e9)
device_h0.operator_separation = True

# Configure an FMCW radar 
radar_h0 = Radar()
radar_h0.waveform = FMCW(bandwidth=bandwidth, num_chirps=1, chirp_duration=2e-6, pulse_rep_interval=2e-6)
radar_h1 = deepcopy(radar_h0)

radar_h0.device = device_h0
radar_h0.reference = device_h0
radar_h1.device = device_h1
radar_h1.reference = device_h1

# Configure a ZP-OFDM JCAS radar
ofdm = OFDMWaveform(subcarrier_spacing=bandwidth/1000, num_subcarriers=1000, oversampling_factor=1)
ofdm.pilot_section = SchmidlCoxPilotSection()
jcas_h0 = MatchedFilterJcas(max_range=radar_h0.waveform.max_range)
jcas_h0.waveform_generator = ofdm
jcas_h1 = deepcopy(jcas_h0)

jcas_h0.device = device_h0
jcas_h0.reference = device_h0
jcas_h1.device = device_h1
jcas_h1.reference = device_h1


# Configure a radar channel
channel_h1 = RadarChannel(target_range=(0.,100.), radar_cross_section = 1., attenuate=False, target_exists=True)
channel_h0 = RadarChannel(target_range=25., radar_cross_section = 1., attenuate=False, target_exists=False)
simulation.scenario.set_channel(device_h1, device_h1, channel_h1)
simulation.scenario.set_channel(device_h0, device_h0, channel_h0)
simulation.scenario.channel(device_h1, device_h0).gain = 0.

# Configure evaluations
simulation.add_evaluator(ReceiverOperatingCharacteristic(radar_h1, radar_h0))
simulation.add_evaluator(ReceiverOperatingCharacteristic(jcas_h1, jcas_h0))

# Run the simulation
simulation.run()

fmcw_transmission = radar_h1.transmit()
jcas_transmission = jcas_h1.transmit()

device_signals = device_h1.transmit()
channel_h1.target_range = 25
propagated_signals, _, _ = channel_h1.propagate(device_signals)
device_h1.receive(propagated_signals, snr=float('inf'))

fmcw_reception = radar_h1.receive()
jcas_reception = jcas_h1.receive()

fmcw_cube = fmcw_reception.cube
jcas_cube = jcas_reception.cube
 
fmcw_cube.normalize_power()
jcas_cube.normalize_power()

savemat(path.join(simulation.results_dir, 'range.mat'), {
   'jcas': np.abs(jcas_cube.data.flatten()),
   'fmcw': np.abs(fmcw_cube.data.flatten()), 
   'jcas_range_bins': jcas_cube.range_bins,
   'fmcw_range_bins': fmcw_cube.range_bins,
})
