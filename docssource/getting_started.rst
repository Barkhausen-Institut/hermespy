.. |br| raw:: html

     <br>

***************
Getting Started
***************

Assuming HermesPy is properly :doc:`installed</installation>` within the currently selected Python environment,
users may define custom wireless communication scenarios to be investigated within the context of
:doc:`Simulations</api/simulation/index>` or :doc:`Hardware Loops</api/hardware_loop/index>`.
The whole HermesPy suite can either be directly integrated into custom software projects and operated as a plug-in
library via a detailed object-oriented programming interface.

This section provides a rough description of the HermesPy software architecture and gives an introduction
into both the library and command line interface in order to get new users quickly accustomed.

=====================
HermesPy Architecture
=====================

In its core, the HermesPy API aims to abstract the process of wireless signal processing
within a strictly object-oriented class structure.
Each processing step is represented by a dedicated class and can be adapted and customized by the software user.

Consider a typically heterogeneous wireless scenario featuring multiple entities transmitting and receiving electromagnetic waveforms.
The physical description of said entities is referred to as a :doc:`Device<api/core/device.Device>` in Hermes.
:doc:`Devices<api/core/device.Device>` provide general information required for the modeling of electromagnetic propagation,
such as carrier frequency, spatial position, orientation, number of available antennas and the respective antenna array topology.
Now, while :doc:`Devices<api/core/device.Device>` describe the physical properties,
the digital signal processing required for generating waveform transmissions and receptions is modeled by :doc:`Transmitters<api/core/device.Transmitter>`
and :doc:`Receivers<api/core/device.Receiver>`, respectively.
They form HermesPy's general abstraction for digital signal processing applied before digital-to-analog conversion and after analog-to-digital conversion, respectively:

.. mermaid::
   :align: center

   graph LR

   subgraph dspi [DSP Transmit Layer]

      oai[Transmitter A]
      obi[Transmitter B]
   end
      
   subgraph phys [Physical Layer]

      dai[Device A]
      dbi[Device B]


      dao[Device A]
      dbo[Device B]
   end

   subgraph dspo [DSP Receive Layer]

      oao[Receiver A]
      obo[Receiver B]
   end

   oai --> dai
   obi --> dbi
   dai --> dao
   dai --> dbo
   dbi --> dbo
   dbi --> dao
   dao --> oao
   dbo --> obo

   click oai "api/core/device.Transmitter.html" "Transmitter"
   click obi "api/core/device.Transmitter.html" "Transmitter"
   click dai "api/core/device.Device.html" "Device"
   click dbi "api/core/device.Device.html" "Device"
   click dao "api/core/device.Device.html" "Device"
   click dbo "api/core/device.Device.html" "Device"
   click oao "api/core/device.Receiver.html" "Receiver"
   click obo "api/core/device.Receiver.html" "Receiver"

A typical information flow consists of a :doc:`Transmitter<api/core/device.Transmitter>` generating a base-band waveform,
submitting it to its assigned :doc:`Device<api/core/device.Device>`,
followed by the :doc:`Device<api/core/device.Device>` emitting the submitted transmission in RF-band, while simultaneously recording impinging broadcasts.
The recorded broadcasts are submitted to the assigend :doc:`Receivers<api/core/device.Receiver>` to be processed.

There are two types of devices, namely :doc:`Simulated<api/simulation/simulated_device>` and :doc:`Physical<api/hardware_loop/physical_device.PhysicalDevice>`,
which both inherit from the abstract :doc:`Device<api/core/device.Device>` base:

.. mermaid::
   :align: center

   classDiagram

   class Device {
      <<Abstract>>
      +transmit() : DeviceTransmisison*
      +receive() : DeviceReception*
   }

   class PhysicalDevice {
      <<Abstract>>
      +transmit() : PhyiscalDeviceTransmisison
      +receive() : PhysicalDeviceReception
   }

   class SimulatedDevice {
      +transmit() : SimulatedDeviceTransmisison
      +receive() : SimulatedDeviceReception
   }

   class Transmitter {

      +transmit() : Transmission
   }

   class Receiver {

      +receive() : Reception
   }

   PhysicalDevice ..|> Device
   SimulatedDevice ..|> Device
   Device *--* Transmitter
   Device *--* Receiver

   link Device "api/core/device.Device.html" "Device"
   link PhysicalDevice "api/hardware_loop.physical_device.PhysicalDevice.html" "Physical Device"
   link SimulatedDevice "api/simulation.simulated_device.html" "Simulated Device"
   link Transmitter "api/core/device.Transmitter.html"
   link Receiver "api/core/device.Receiver.html"

Depending on which :doc:`Device<api/core/device.Device>` realization is selected,
Hermes acts as  either a physical layer simulation platform or a hardware testbed,
with the advantage that implemented signal processing algorithms, which are simply classes inheriting from
either :doc:`Transmitter<api/core/device.Transmitter>`, :doc:`Receiver<api/core/device.Receiver>`, or both,
integrate seamlessly into both simulation and hardware testbed setups over a unifying API without the need for any code modifications.
Three types of signal processing pipelines are currently provided by Hermes out of the box and shipped in individual namespace packages:

* :doc:`Modems<api/modem/index>` provide the typical physical layer signal processing pipeline for digital communication systems, including mapping, modulation, forward error corretion,
  precoding / beamforming, synchronization, channel estimation and channel equalization.
* :doc:`Radars<api/radar/index>` provide the typical physical layer signal processing pipeline for sensing systems, including beamforming and target detection.
* :doc:`JCAS<api/jcas/index>` is a combination of both, providing a physical layer signal processing pipeline for joint communication and sensing systems.

The following subsections will introduce how to set up simulations and run hardware testbed setups.

============
Simulations
============

:doc:`Simulation<api/simulation/index>` campaigns are defined by a set of :doc:`SimulatedDevices<api/simulation/simulated_device>` interconnected by :doc:`Channel Models<api/channel/channel.Channel>`,
with the combination of both forming a :doc:`SimulationScenario<api/simulation/scenario.SimulationScenario>`.
Therefore, considering channel reciprocity, a :doc:`SimulationScenario<api/simulation/scenario.SimulationScenario>` featuring :math:`D` devices requires the specification of :math:`\tfrac{D(D+1)}{2}` :doc:`Channel Models<api/channel/channel.Channel>`.
Considering a model featuring :math:`D=2` dedicated devices, the following simulated physical layer model is formed:

.. include:: flowcharts/simulated_physical_layer.rst

Each device is linked to two channel instances.
Note that, even though four channels are depicted, channel `B, A` links to the same channel instance as `A, B` due to the reciprocity assumption,
leading to a total of :math:`\tfrac{2(2+1)}{2}=3` unique channel instances for the depicted scenario.
Initializing said scenario is as simple as creating a new simulation instance and adding two
devices: 

.. literalinclude:: ../_examples/getting_started/simulation.py
   :language: python
   :linenos:
   :lines: 08-13

When adding new devices to a simulation, the simulation will automatically intialize the required channel instances
as :doc:`IdealChannels<api/channel/ideal>`.
However, the user may freely select from a multitude of different channel models, which are provided by the :doc:`Channel<api/channel/index>` package.
For example, the following snippet configures the depicted scenario with a :doc:`5G Tapped Delay Line Channel<api/channel/fading/tdl>`:

.. literalinclude:: ../_examples/getting_started/simulation.py
   :language: python
   :linenos:
   :lines: 19-20

While :doc:`SimulatedDevices<api/simulation/simulated_device>` and :doc:`Channel Models<api/channel/channel.Channel>` form the physical layer description
of a simulation, the signal processing, i.e. the transmit and receive layer, generates the waveforms that will actually be generated by devices and transmitted
over the channels.
For communication cases in which we want to declare one device as the sole transmitter and one device as the sole receiver,
HermesPy offers the :doc:`SimplexLink<api/modem/modem.SimplexLink>` class, which automatically configures the transmit and receive layer of the devices:

.. literalinclude:: ../_examples/getting_started/simulation.py
   :language: python
   :linenos:
   :lines: 22-25

The :doc:`Modem<api/modem/index>` package provides a range of communication waveform implementations,
for this minimal introduction we will choose a :doc:`Root-Raised-Cosine<api/modem/waveform.single_carrier.RootRaisedCosine>` single carrier waveform:

.. literalinclude:: ../_examples/getting_started/simulation.py
   :language: python
   :linenos:
   :lines: 27-32

We may now already directly call the :doc:`SimplexLink<api/modem/modem.SimplexLink>`'s transmit and receive rountines to directly generate, process and visualize
the generated information such as base-band waveforms and symbol constellations:

.. literalinclude:: ../_examples/getting_started/simulation.py
   :language: python
   :linenos:
   :lines: 36-42

This will bypass physical layer simulations including device and channel models and directly receive the transmitted waveform, resulting in perfect information recovery.
Refering back to the intial architecture graph, we patched the transmit layer directly into the receive layer.

During simulations, however, the full physical layer is considered.
HermesPy is drop-based, meaning with each call of the :doc:`Simulation<api/simulation/simulation.Simulation>`'s drop method,
new realizations of the configured channel models are generated, the transmit routines of all :doc:`Transmitters<api/core/device.Transmitter>` are called,
the generated waveforms are propagated over the configured channels and the receive routines of all :doc:`Receivers<api/core/device.Receiver>`.
The generated information is collected in :doc:`SimulatedDrops<api/simulation/drop.SimulatedDrop>` to be accessed by the user:

.. literalinclude:: ../_examples/getting_started/simulation.py
   :language: python
   :linenos:
   :lines: 44-47

After the generation of a new :doc:`SimulatedDrop<api/simulation/drop.SimulatedDrop>`,
:class:`Evaluators<hermespy.core.pymonte.evaluation.Evaluator>` may be used to conveniently extract performance information.
For instance, the bit error rate of the generated drop may be extracted by a :doc:`BitErrorEvaluator<api/modem/evaluators.ber>`:

.. literalinclude:: ../_examples/getting_started/simulation.py
   :language: python
   :linenos:
   :lines: 49-50

This is the core routine of a typical Monte Carlo simulation, which is usually conducted over a grid of parameter values.
For each parameter combination, a new :doc:`SimulatedDrop<api/simulation/drop.SimulatedDrop>` is generated and evaluated.
This process is executed multiple times in parallel, depending on the number of available CPU cores and the user's configuration.
Finally, the generated evaluations are concatenated towards a single result.

.. include:: flowcharts/simulation_parallel_monte_carlo.rst

A simulation iterating over the receiving device's signal to noise ratio as parameters and estimating the respective bit error RootRaisedCosine
can be launched by executing

.. literalinclude:: ../_examples/getting_started/simulation.py
   :language: python
   :linenos:
   :lines: 52-57

which will result in a rendered plot being generated.
The full code snippet implementing the above introduction can be downloaded from `GitHub - Getting Started Simulation <https://github.com/Barkhausen-Institut/hermespy/blob/main/_examples/getting_started/simulation.py>`_.
For more complex simulation examples and instructions on how to integrate and evaluate your own signal processing algorithms in HermesPy,
please refer to the :doc:`Tutorials <tutorials>` section.

=============
Hardware Loop
=============

:doc:`Hardware Loops<api/hardware_loop/index>` allow for the execution and measurement collection of signal processing algorithms in the transmit and receive processing layer
on real hardware.
They are defined by a set of :doc:`PhysicalDevices<api/hardware_loop/physical_device.PhysicalDevice>` forming a :doc:`PhysicalScenario<api/hardware_loop/scenario.PhysicalScenario>`,
which represents the physical layer description of the hardware loop.

.. mermaid::
   :align: center

   graph LR
    
   subgraph phys [Hardware Loop Physical Layer]
      direction LR
      dai[PhysicalDevice A]
      dbi[PhysicalDevice B]

      world((Real World))

      dao[PhysicalDevice A]
      dbo[PhysicalDevice B]
   end

   dai -.-> world -.-> dao
   dai -.-> world -.-> dbo
   dbi -.-> world -.-> dbo
   dbi -.-> world -.-> dao

   click dai "api/hardware_loop.physical_device.PhysicalDevice.html" "Physical Device"
   click dbi "api/hardware_loop.physical_device.PhysicalDevice.html" "Physical Device"
   click dao "api/hardware_loop.physical_device.PhysicalDevice.html" "Physical Device"
   click dbo "api/hardware_loop.physical_device.PhysicalDevice.html" "Physical Device"

When compared to simulations, :doc:`Hardware Loops<api/hardware_loop/index>` obvisouly lack channel and hardware modeling capabilities.
Instead, each trigger of a :doc:`PhysicalDevice<api/hardware_loop/physical_device.PhysicalDevice>` will generate a transmission and transmit the
respective base-band samples over the air.
In other words, the simulated device and channel models have been replaced by the real world.

Setting up a :doc:`Hardware Loop<api/hardware_loop/index>` is as simple as creating a new :doc:`PhysicalScenario<api/hardware_loop/scenario.PhysicalScenario>`
and passing it to a new :doc:`HardwareLoop<api/hardware_loop/hardware_loop.HardwareLoop>` instance:

.. literalinclude:: ../_examples/getting_started/loop.py
   :language: python
   :linenos:
   :lines: 07-13

The :doc:`PhysicalScenarioDummy<api/hardware_loop/physical_device_dummy.PhysicalScenarioDummy>` is a physical scenario implementation intended for testing and demonstration
purposes and does not require real hardware.
Instead, the :doc:`PhysicalDeviceDummies<api/hardware_loop/physical_device_dummy.PhysicalDeviceDummy>` instances behave identical to :doc:`SimulatedDevices<api/simulation/simulated_device>`.
For this reason, we can also assign channel models to the managing :doc:`PhysicalScenarioDummy<api/hardware_loop/physical_device_dummy.PhysicalScenarioDummy>` instances:

.. literalinclude:: ../_examples/getting_started/loop.py
   :language: python
   :linenos:
   :lines: 15-17

This is, of course, not possible in real hardware scenarios such as 
:doc:`USRP Systems<api/hardware_loop/uhd.system.UsrpSystem>` featuring :doc:`USRP Devices<api/hardware_loop/uhd.usrp.UsrpDevice>` or
:doc:`Audio Scenarios<api/hardware_loop/audio.scenario.AudioScenario>` featuring :doc:`Audio Devices<api/hardware_loop/audio.device.AudioDevice>`.

For communication cases in which we want to declare one device as the sole transmitter and one device as the sole receiver,
HermesPy offers the :doc:`SimplexLink<api/modem/modem.SimplexLink>` class, which automatically configures the transmit and receive layer of the devices:

.. literalinclude:: ../_examples/getting_started/loop.py
   :language: python
   :linenos:
   :lines: 19-22

The :doc:`Modem<api/modem/index>` package provides a range of communication waveform implementations,
for this minimal introduction we will choose a :doc:`Root-Raised-Cosine<api/modem/waveform.single_carrier.RootRaisedCosine>` single carrier waveform:

.. literalinclude:: ../_examples/getting_started/loop.py
   :language: python
   :linenos:
   :lines: 24-29

Just like the simulation pipeline, the hardware loop runtime will generate drops to be evaluated.
However, instead of multiple drops being generated in parallel, the hardware loop's drop generation is performed sequentially by triggering
the configured :doc:`PhysicalDevices<api/hardware_loop/physical_device.PhysicalDevice>`.

After the generation of a new :doc:`Drop<api/core/drop>`,
:class:`Evaluators<hermespy.core.pymonte.evaluation.Evaluator>` may be used to conveniently extract performance information.
For instance, the bit error rate of the generated drop may be extracted by a :doc:`BitErrorEvaluator<api/modem/evaluators.ber>`:

.. literalinclude:: ../_examples/getting_started/loop.py
   :language: python
   :linenos:
   :lines: 31-33

Working with real hardware usually requires a lot of oversight and debugging,
so the :doc:`HardwareLoop<api/hardware_loop/hardware_loop.HardwareLoop>` features a visualization
interface which will render plots of required information in real-time:

.. literalinclude:: ../_examples/getting_started/loop.py
   :language: python
   :linenos:
   :lines: 35-38

The plots will be updated with each new :doc:`Drop<api/core/drop>`.
An overview of existing visualization routines can be found in :doc:`Visualizers<api/hardware_loop/visualizers>`.

Identically to the simulation pipeline, the :doc:`HardwareLoop<api/hardware_loop/hardware_loop.HardwareLoop>`
can be configured to iterate over a grid of parameter values and generate a fixed number of drops per parameter combination:

.. literalinclude:: ../_examples/getting_started/loop.py
   :language: python
   :linenos:
   :lines: 40-42

Setting the `results_dir` parameter will result in a consolidation of all drop data and evaluations into a single `drops.h5` file within the respective directory.
The data can be accessed by the user for further processing, or even directly replayed by the :doc:`HardwareLoop<api/hardware_loop/hardware_loop.HardwareLoop>`:

.. literalinclude:: ../_examples/getting_started/loop.py
   :language: python
   :linenos:
   :lines: 44-45

The full code snippet implementing the above introduction can be downloaded from `GitHub - Getting Started Loop <https://github.com/Barkhausen-Institut/hermespy/blob/main/_examples/getting_started/loop.py>`_.
For more complex simulation examples and instructions on how to integrate and evaluate your own signal processing algorithms in HermesPy,
please refer to the :doc:`Tutorials <tutorials>` section.
