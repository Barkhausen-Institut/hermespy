========
Devices
========

.. inheritance-diagram:: hermespy.simulation.simulated_device.SimulatedDevice
   :top-classes: hermespy.core.device.Device,hermespy.core.transformation.Transformable
   :parts: 1

:class:`SimulatedDevices<hermespy.simulation.simulated_device.SimulatedDevice>` are the description of any entity
capable of transmitting or receiving electromagnetic waves within a HermesPy simulation scenario.
A simulation scenario may feature one or multiple devices freely exchanging electromagnetic waves at
any different carrier frequencies and bandwidths.

Creating a new simulated device is as simple as initializing a new class instance,
or calling :meth:`new_device<hermespy.core.pipeline.Pipeline.new_device>` on a :class:`Simulation<hermespy.simulation.simulation.Simulation>` instance.

.. literalinclude:: ../scripts/examples/simulation_SimulatedDevice.py
   :language: python
   :linenos:
   :lines: 18-23

There exist a number of attributes that can be configured to describe the device's physical properties,
such as its antenna front-end, its radio-frequency chain, the isolation / leakage between transmit and receive chains,
the mutual coupling between multiple antennas, the synchronizatio and the hardware noise.

.. literalinclude:: ../scripts/examples/simulation_SimulatedDevice.py
   :language: python
   :linenos:
   :lines: 25-50

Additionally, within the context of a simulation scenario,
users may configure the device's position, orientation and velocity.

.. literalinclude:: ../scripts/examples/simulation_SimulatedDevice.py
   :language: python
   :linenos:
   :lines: 52-57

Note that these properties are only considered by spatial channel models linking to the respective devices,
and are otherwise ignored by the simulation engine.

By themselves, devices neither generate any waveforms, nor perform any signal processing on 
received waveforms.
Instead, transmit and receive signal processing algorithms have to be assigned to the device's respective
:attr:`transmitters<hermespy.core.device.Device.transmitters>` and :attr:`receivers<hermespy.core.device.Device.receivers>`
slots.

.. literalinclude:: ../scripts/examples/simulation_SimulatedDevice.py
   :language: python
   :linenos:
   :lines: 59-72

The selected singnal processing algorithms in the snippet above are rather simplistic and only
generate a static signal that is continuously transmitted, and receive a fixed number of samples, respectively.
More complex algorithms implementing communication modems, radars or joint communication and sensing algorithms
are available in HermesPy's :doc:`modem`, :doc:`radar` and :doc:`jcas` modules.

Outside of full-fledged Monte Carlo simulations, users may inspect the output of a configured simulated device
by calling the :meth:`transmit<hermespy.simulation.simulated_device.SimulatedDevice.transmit>` method.

.. literalinclude:: ../scripts/examples/simulation_SimulatedDevice.py
   :language: python
   :linenos:
   :lines: 74-75

Similarly, users may inspect the signal processing result of configured receive signal processing algorithms
by calling the :meth:`receive<hermespy.simulation.simulated_device.SimulatedDevice.receive>` method
and providing a :class:`Signal<hermespy.core.signal_model.Signal>` model of the assumed waveform impinging onto the device.

.. literalinclude:: ../scripts/examples/simulation_SimulatedDevice.py
   :language: python
   :linenos:
   :lines: 77-84

The :meth:`transmit<hermespy.simulation.simulated_device.SimulatedDevice.transmit>` routine is a wrapper around multiple subroutines,
that are individually executed during simulation runtime for performance optimization reasons, finally returning a
:class:`SimulatedDeviceTransmission<hermespy.simulation.simulated_device.SimulatedDeviceTransmission>` dataclass containing
all information generated during the simulation of the device's transmit chain.

.. mermaid::
   :align: center

   graph TD
      
   transmit_operators[transmit_operators]
   operator_transmissions{{List Transmission}}
   generate_output[generate_output]
   output{{SimulatedDeviceOutput}}
   deviceTransmission{{SimulatedDeviceTransmission}}
   triggerRealization{{TriggerRealization}}
   
   transmit_operators --> operator_transmissions --> generate_output --> output
   operator_transmissions --> deviceTransmission
   output --> deviceTransmission
   triggerRealization --> generate_output

   click transmit_operators "core.device.Device.html#hermespy.core.device.Device.transmit_operators" "transmit_operators"
   click operator_transmissions "core.device.Transmission.html#hermespy.core.device.Transmission" "Transmission"
   click generate_output "#hermespy.simulation.simulated_device.SimulatedDevice.generate_output" "generate_output"
   click output "simulation.simulated_device.SimulatedDeviceOutput.html" "SimulatedDeviceOutput"
   click deviceTransmission "simulation.simulated_device.SimulatedDeviceTransmission.html" "SimulatedDeviceTransmission"
   click triggerRealization "simulation.synchronization.TriggerRealization.html" "TriggerRealization"

The :meth:`receive<hermespy.simulation.simulated_device.SimulatedDevice.receive>` method requires the input of the signal to be processed by the device,
and is also a wrapper around multiple subroutines, that are individually executed during simulation runtime for performance optimization reasons,
finally returning a :class:`SimulatedDeviceReception<hermespy.simulation.simulated_device.SimulatedDeviceReception>` dataclass containing
all information generated during the simulation of the device's receive chain.

.. mermaid::
   :align: center

   graph TD
   
   signal{{Impinging Signal}}
   triggerRealization{{TriggerRealization}}
   leaking_signal{{Leaking Signal}}
   processed_input{{ProcessedSimulatedDeviceInput}}
   receptions{{List Reception}}
   device_reception{{SimulatedDeviceReception}}
   
   subgraph process_input [process_input]
   
       realize_reception[[realize_reception]]
       process_from_realization[[process_from_realization]]
       device_realization{{SimulatedDeviceReceiveRealization}}
   end
   
   receive_operators[receive_operators]
   
   signal --> process_input
   triggerRealization --> process_input
   leaking_signal --> process_input
   realize_reception --> device_realization --> process_from_realization
   process_from_realization --> processed_input
   processed_input --> receive_operators
   receive_operators --> receptions
   receptions --> device_reception
   processed_input --> device_reception

   click signal "core.signal_model.Signal.html" "Signal"
   click triggerRealization "simulation.synchronization.TriggerRealization.html" "TriggerRealization"
   click leaking_signal "core.signal_model.Signal.html" "Signal"
   click realize_reception "#hermespy.simulation.simulated_device.SimulatedDevice.realize_reception" "realize_reception"
   click device_realization "simulation.simulated_device.SimulatedDeviceReceiveRealization.html" "SimulatedDeviceReceiveRealization"
   click processed_input "simulation.simulated_device.ProcessedSimulatedDeviceInput.html" "ProcessedSimulatedDeviceInput"
   click receive_operators "core.device.Device.html#hermespy.core.device.Device.receive_operators" "receive_operators"
   click receptions "core.device.Reception.html#hermespy.core.device.Reception" "Reception"
   click device_reception "simulation.simulated_device.SimulatedDeviceReception.html" "SimulatedDeviceReception"
   click process_from_realization "#hermespy.simulation.simulated_device.SimulatedDevice.process_from_realization" "process_from_realization"

.. toctree::
   :hidden:

   simulation.simulated_device.ProcessedSimulatedDeviceInput
   simulation.simulated_device.SimulatedDeviceOutput
   simulation.simulated_device.SimulatedDeviceReceiveRealization
   simulation.simulated_device.SimulatedDeviceReception
   simulation.simulated_device.SimulatedDeviceTransmission

.. autoclass:: hermespy.simulation.simulated_device.SimulatedDevice

.. footbibliography::
