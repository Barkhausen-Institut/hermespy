import numpy as np
from scipy.constants import pi
from scipy.io import savemat, loadmat

from hermespy.simulation import Simulation
from hermespy.channel import MultipathFading5GTDL
from hermespy.modem import BitErrorEvaluator, Modem
from hermespy.modem.waveform_generator_psk_qam import RootRaisedCosine, PskQamLeastSquaresChannelEstimation, PskQamZeroForcingChannelEqualization
from hermespy.tools import db2lin
from hermespy.core import ConsoleMode

simulation = Simulation(console_mode=ConsoleMode.LINEAR, ray_address='auto')
device = simulation.scenario.new_device(carrier_frequency=3.7e9)
simulation.scenario.set_channel(device, device, MultipathFading5GTDL())

modem = Modem()
modem.device = device
modem.waveform_generator = RootRaisedCosine(oversampling_factor=4, num_preamble_symbols=16, num_data_symbols=100, symbol_rate=100e6, modulation_order=16)
modem.waveform_generator.channel_estimation = PskQamLeastSquaresChannelEstimation()
modem.waveform_generator.channel_equalization = PskQamZeroForcingChannelEqualization()

simulation.new_dimension('snr', [db2lin(x) for x in np.arange(-10, 20, .5)])
simulation.add_evaluator(BitErrorEvaluator(modem, modem))
simulation.num_samples, simulation.min_num_samples = 10000, 10000
simulation.plot_results = False

max_num_actors = 1000
step = 10
num_trials = 50
time_table = -np.ones((int(max_num_actors / step), num_trials), dtype=float)
savepath = '/home/jan.adler/paper/scaling/results.mat'

# Make sure the result can be saved
#savemat(savepath, {'times': time_table})
time_table = loadmat(savepath)['times']

# Run a first simulation to account for enviroment setups
_ = simulation.run()

for n in range(1, 1 + max_num_actors, step):
    
    simulation.num_actors = n
    
    for t in range(num_trials):
        
        if time_table[n-1, t] > 0:
            continue
        
        result = simulation.run()
        time_table[n-1, t] = result.performance_time
        
        savemat(savepath, {'times': time_table})
