from scipy.constants import speed_of_light

sampling_rate = 1e6
carrier_frequency = 70e9
wavelength = speed_of_light / carrier_frequency

# Initialize a new simulation
from hermespy.simulation import Simulation
simulation = Simulation(seed=42)

# Create a new device and assign it the antenna array
from hermespy.simulation import SimulatedIdealAntenna, SimulatedUniformArray
from hermespy.core import  Transformation
import numpy as np

base_station_device = simulation.new_device(
    antennas=SimulatedUniformArray(SimulatedIdealAntenna, .5 * wavelength, (8, 8, 1)),
    carrier_frequency=carrier_frequency,
    pose=Transformation.From_Translation(np.array([0, 0, 0])),
)

# Configure a probing signal to be transmitted.
from hermespy.modem import RootRaisedCosineWaveform, SingleCarrierLeastSquaresChannelEstimation, SingleCarrierZeroForcingChannelEqualization, SingleCarrierCorrelationSynchronization

waveform = RootRaisedCosineWaveform(
    symbol_rate=sampling_rate//2,
    oversampling_factor=2,
    num_preamble_symbols=32,
    num_data_symbols=128,
    roll_off=.9,
)
waveform.synchronization = SingleCarrierCorrelationSynchronization()
waveform.channel_estimation = SingleCarrierLeastSquaresChannelEstimation()
waveform.channel_equalization = SingleCarrierZeroForcingChannelEqualization()

# Configure the base station device to receive the beamformed probing signal
from hermespy.beamforming import ConventionalBeamformer
from hermespy.modem import ReceivingModem
from copy import deepcopy

beamformer = ConventionalBeamformer()
base_station_device.receive_coding[0] = beamformer

base_station_receiver = ReceivingModem(waveform=deepcopy(waveform))
base_station_device.add_dsp(base_station_receiver)


# Create two simulated devices representing the user equipments
user_equipment_device_1 = simulation.new_device(
    carrier_frequency=carrier_frequency,
    pose=Transformation.From_Translation(np.array([200., 200., 100.])),
)
user_equipment_device_2 = simulation.new_device(
    carrier_frequency=carrier_frequency,
    pose=Transformation.From_Translation(np.array([200., -200., 100.])),
)

# Configure the user equipments to transmit the signal
from hermespy.modem import TransmittingModem

user_equipment_transmitter_1 = TransmittingModem(waveform=deepcopy(waveform))
user_equipment_transmitter_2 = TransmittingModem(waveform=deepcopy(waveform))
user_equipment_device_1.add_dsp(user_equipment_transmitter_1)
user_equipment_device_2.add_dsp(user_equipment_transmitter_2)


# Focus the base station's main lobe on the desired user equipment.
from hermespy.simulation import DeviceFocus

beamformer.receive_focus = [
    DeviceFocus(user_equipment_device_1), # Focus on User Equipmment 1
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

# Run the simulation and the inspect the received signal quality from the respective UEs.
from hermespy.modem import ConstellationEVM

simulation.add_evaluator(ConstellationEVM(user_equipment_transmitter_1, base_station_receiver))
simulation.add_evaluator(ConstellationEVM(user_equipment_transmitter_2, base_station_receiver))


# Creating a new dimension to dynamically switch the focus of the beamformer during the simulation campaign.
simulation.new_dimension(
    'focused_device',
    [user_equipment_device_1, user_equipment_device_2],
    beamformer.receive_focus[0]
)

result = simulation.run()
