from scipy.constants import speed_of_light

carrier_frequency = 70e9
wavelength = speed_of_light / carrier_frequency

# Initialize a new simulation
from hermespy.simulation import Simulation
simulation = Simulation(seed=42)

# Create a new device and assign it the antenna array
from hermespy.simulation import SimulatedIdealAntenna, SimulatedUniformArray
from hermespy.core import Transformation
import numpy as np

base_station_device = simulation.new_device(
    antennas=SimulatedUniformArray(SimulatedIdealAntenna, .5 * wavelength, (4, 4, 1)),
    carrier_frequency=carrier_frequency,
    pose=Transformation.From_Translation(np.array([0, 0, 0])),
)

# Configure a probing signal to be transmitted from the base station
from hermespy.modem import RootRaisedCosineWaveform, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization, SingleCarrierCorrelationSynchronization

waveform = RootRaisedCosineWaveform(
    num_preamble_symbols=32,
    num_data_symbols=128,
    roll_off=.9,
)
waveform.synchronization = SingleCarrierCorrelationSynchronization()
waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

# Configure the base station device to transmit the beamformed probing signal
from hermespy.beamforming import ConventionalBeamformer
from hermespy.modem import TransmittingModem
from copy import deepcopy

beamformer = ConventionalBeamformer()
base_station_device.transmit_coding[0] = beamformer

base_station_transmitter = TransmittingModem(waveform=deepcopy(waveform))
base_station_device.add_dsp(base_station_transmitter)

# Create Two simulated device representing the user equipments
user_equipment_device_1 = simulation.new_device(
    carrier_frequency=carrier_frequency,
    pose=Transformation.From_Translation(np.array([100., 100., 100.])),
)
user_equipment_device_2 = simulation.new_device(
    carrier_frequency=carrier_frequency,
    pose=Transformation.From_Translation(np.array([100., -100., 100.])),
)

# Configure the user equipments to receive the signal
from hermespy.modem import ReceivingModem

user_equipment_receiver_1 = ReceivingModem(waveform=deepcopy(waveform))
user_equipment_receiver_2 = ReceivingModem(waveform=deepcopy(waveform))
user_equipment_device_1.add_dsp(user_equipment_receiver_1)
user_equipment_device_2.add_dsp(user_equipment_receiver_2)

# Focus the base station's main lobe on the desired user equipment and nulls on the others
from hermespy.simulation import DeviceFocus

beamformer.transmit_focus = [
    DeviceFocus(user_equipment_device_1),  # Focus on User Equipmment 1
]

# Configure a channel between base station and the UEs
from hermespy.channel import SpatialDelayChannel

simulation.set_channel(
    base_station_device,
    user_equipment_device_1,
    SpatialDelayChannel(model_propagation_loss=False),
)

simulation.set_channel(
    base_station_device,
    user_equipment_device_2,
    SpatialDelayChannel(model_propagation_loss=False),
)

# Run the simulation and the evaluate the received power from the respective UEs.
from hermespy.core import ReceivePowerEvaluator

simulation.add_evaluator(ReceivePowerEvaluator(user_equipment_receiver_1))
simulation.add_evaluator(ReceivePowerEvaluator(user_equipment_receiver_2))

# Creating a new dimension to dynamically switch the focus of the beamformer during the simulation campaign.
simulation.new_dimension(
    'focused_device',
    [user_equipment_device_1, user_equipment_device_2],
    beamformer.transmit_focus[0]
)

result = simulation.run()
