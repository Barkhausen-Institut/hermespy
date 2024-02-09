# -*- coding: utf-8 -*-

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


import matplotlib.pyplot as plt

from hermespy.radar import Radar, FMCW, ThresholdDetector
from hermespy.simulation import Simulation
from hermespy.channel import SingleTargetRadarChannel

# Configure an FMCW radar with a threshold detector
radar = Radar()
radar.waveform = FMCW()
radar.detector = ThresholdDetector(.5)

# Initialize a simulation and configure a radar channel
simulation = Simulation()
radar.device = simulation.new_device(carrier_frequency=60e9)
channel = SingleTargetRadarChannel(.5 * radar.max_range, 1.)
simulation.scenario.set_channel(radar.device, radar.device, channel)

# Generate a single simulation drop
simulation.scenario.drop()

# Visualizue the generated radar information
radar.transmission.signal.plot(title='Transmitted Radar Signal')
radar.reception.signal.plot(title='Received Radar Signal')
radar.reception.cube.plot_range(title='Range Power Profile')
radar.reception.cube.plot_range_velocity(title='Range Velocity Map')
radar.reception.cloud.visualize(title='Radar Point Cloud')

plt.show()
