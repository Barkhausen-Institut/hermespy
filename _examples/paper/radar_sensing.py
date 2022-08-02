import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from hermespy.channel import RadarChannel
from hermespy.modem.waveform_generator_ofdm import SchmidlCoxPilotSection
from hermespy.radar import Radar, FMCW
from hermespy.jcas import MatchedFilterJcas
from hermespy.simulation import SimulatedDevice
from hermespy.modem import WaveformGeneratorOfdm


device = SimulatedDevice(carrier_frequency=70e9)
device.operator_separation = True
channel = RadarChannel(target_range = 25, radar_cross_section = 1., transmitter=device, receiver=device)

radar = Radar()
radar.device = device
radar.waveform = FMCW(max_range=50, bandwidth=1e9, sampling_rate=1e9, num_chirps=1)
radar.transmit()

ofdm = WaveformGeneratorOfdm(subcarrier_spacing=1e6, num_subcarriers=1000)
ofdm.pilot_section = SchmidlCoxPilotSection()
jcas = MatchedFilterJcas(max_range=50)
jcas.waveform_generator = ofdm
jcas.device = device
jcas.transmit()

device_signals = device.transmit()
propagated_signals, _, _ = channel.propagate(device_signals)
device.receive(propagated_signals)

fmcw_cube, _ = radar.receive()
_, _, _, jcas_cube, _ = jcas.receive()
jcas_cube.normalize_power()
fmcw_cube.normalize_power()

savemat('D:\\hermes_paper\\radar_sensing\\range.mat', {
   'jcas': np.abs(jcas_cube.data.flatten()),
   'fmcw': np.abs(fmcw_cube.data.flatten()), 
   'range_bins': fmcw_cube.range_bins,
})
