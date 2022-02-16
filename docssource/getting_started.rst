.. |br| raw:: html

     <br>

***************
Getting Started
***************

Assuming HermesPy is properly :doc:`installed</installation>` within the currently selected Python environment,
users may define custom wireless communication scenarios to be investigated within the context of
:doc:`Simulations</api/simulation>` or :doc:`Hardware Loops</api/hardware_loop>`.
The whole HermesPy suite can either be directly integrated into custom software projects and operated as a plug-in
library via a detailed object-oriented programming interface or configured by YAML-style configuration files
and launched from any system command line.

This section provides a rough description of the HermesPy software architecture and gives an introduction
into both the library and command line interface in order to get new users quickly accustomed.

=====================
HermesPy Architecture
=====================

In its core, the HermesPy API aims to abstract the process of wireless communication and sensing signal processing
within a strictly object-oriented class structure.
Each processing step is represented by a dedicated class and can be adapted and customized by the software user.

Consider a single link between a receiving and transmitting wireless :doc:`Device</api/core.device>`.
HermesPy does not natively distinguish between Up- Down- and Side-Link,
instead every link between two spatially separated wireless entities is characterized by
two :doc:`Device</api/core.device>` instances and a :doc:`Channel</api/channel.channel>`,
as visualized in the following flowchart:


.. mermaid::
   :align: center

   %%{init: {'theme': 'dark'}}%%
   flowchart LR

   channel{Channel Model}

   subgraph devicea[SimulatedDevice]

       direction TB
       deviceatx>Tx Slot]
       devicearx>Rx Slot]

   end

   subgraph deviceb[SimulatedDevice]

       direction TB
       devicebtx>Tx Slot]
       devicebrx>Rx Slot]
   end

   deviceatx --> channel --> devicearx
   devicebtx --> channel --> devicebrx

Currently two types of devices are supported,
namely :doc:`Simulated Devices</api/simulation.simulated_device>` and
:doc:`Physical Devices</api/hardware_loop.physical_device>`, used within simulation and hardware
verification contexts, respectively.
For the scope of this introduction we will focus on simulated devices, since they, as the name suggests,
do not require any additional hardware from the user.
Complex wireless :doc:`Scenarios</api/core.scenario/>` can theoretically be configured to feature
an unlimited amount of devices.
Within :doc:`Simulations</api/simulation.simulation/>`,
the devices and the channels linking them form a symmetric matrix of channel instances:

.. list-table:: Channel Matrix
   :header-rows: 1
   :stub-columns: 1

   * -
     - Device #1
     - Device #2
     - ...
     - Device #N

   * - Device #1
     - Channel Model (1, 1)
     - Channel Model (1, 2)
     - ...
     - Channel Model (1, N)

   * - Device #2
     - Channel Model (1, 2)
     - Channel Model (2, 2)
     - ...
     - Channel Model (2, N)

   * - ...
     - ...
     - ...
     - ...
     - ...

   * - Device #N
     - Channel Model (1, N)
     - Channel Model (2, N)
     - ...
     - Channel Model (N, N)

Each link channel model may be configured according to the scenario assumptions.
Note that the diagonal of this channel matrix approach patches the devices transmission back as receptions,
enabling, for example, self-interference or sensing investigations.
Currently available channel models are provided by the :doc:`Channel</api/channel>` package.

Each device may transmit and arbitrary :doc:`Signal Model</api/core.signal_model>` over its transmit slot and
receive an arbitrary signal over its receive slot after propagation.
:doc:`Signal Models</api/core.signal_model>` contain base-band samples of the signals transmitted / received by each
device antenna as well as meta-information about the assumed radio-frequency band center frequency and sampling rate.
In general, an unlimited amount of :class:`Operators<hermespy.core.device.Operator>` may be configured to operate on any
device's slots.
Transmit operators may submit individual :doc:`Signal Models</api/core.signal_model>` to its configured device slot.
The signal transmitted by the device will then be formed by a superposition of all submitted operator signals.
Inversely, receive operators are provided with the signal received by its configured device after propagation.
Currently two types of :class:`Duplex Operators<hermespy.core.device.DuplexOperator>`,
operating both the transmit and receive slot of their configured device, are implemented:

* :doc:`Communication Modems</api/modem.modem>` for information exchange in form of bits
* :doc:`Radars</api/radar.radar>` for wireless sensing

These operators each model the sequential signal processing steps for the transmission and reception of their
respective waveforms in a modular fashion.
Each processing step is represented by a customizable or interchangeable class slot.
The :doc:`Communication Modem</api/modem.modem>` operator class currently considers

* :doc:`Bit Sources</api/modem.bits_source>` as the source of data bits to be transmitted
* :doc:`Channel Codings</api/coding.coding>` as the channel coding configuration
* :doc:`Waveform Generators</api/modem.waveform_generator>` as the transmit waveform configuration
* :doc:`Channel Precodings</api/precoding.precoding>` as the channel precoding configuration

while the :doc:`Radar</api/radar.radar>` operator only considers

* :doc:`Radar Waveforms</api/radar.radar>` as the transmit waveform configuration

making it much easier to configure.


.. _GettingStarted_Library:

========
Library
========

This chapter provides several examples outlining the utilization of HermesPy as a library within custom Python projects.
A full description of the application programming interface can be found in the section :doc:`/api/api`.

-------------
Transmissions
-------------

The following code generates the samples of a single communication frame
transmitted by a PSK/QAM modem:

.. literalinclude:: /../_examples/library/getting_started.py
   :language: python
   :linenos:

Within this snippet, multiple statements lead to the generation and simulation of a single communication frame signal.

* Initially, the required Python modules are imported *(lines 1-4)*.
* A new modem operator instance is created *(line 6)*.
* The waveform to be generated by the modem is configured by assigning a specific
  :doc:`Waveform Generator </api/modem.waveform_generator>` instance to the modem's
  waveform_generator property *(line 7)*. |br|
  In our case, this is an instance of a :doc:`PKS/QAM </api/modem.waveform_generator_psk_qam>`
  waveform.
* The device on which the modem operates is defined *(line 9)*.
* An signal model, encoding a single communication frame, emitted by the modem operator
  is generated and plotted *(lines 10-13)*

Executing the snippet will result in a plot similar to

.. image:: images/getting_started_signal_plot.png
  :alt: PSK/QAM default waveform plot

which visualizes the generated samples in time-domain (left sub-plot) and its respective
discrete fourier transform (right sub-plot).

While this is only a minimal example, it highlights the philosophy behind the HermesPy API, namely that
each signal processing step is represented by a class modeling its functionality.
Instances of those classes are assigned to property slots, where they will be executed sequentially
during signal simulation.
Changing the waveform generated by the transmitter defined in the previous snippet
is therefore as simple as assigning a different type of
:doc:`Waveform Generator </api/modem.waveform_generator>`
to its :meth:`hermespy.modem.modem.Modem.waveform_generator` property slot.

Of course, a multitude of parameters can be configured to customize the behaviour of each processing step.
For instance, the frame generated by a :doc:`PKS/QAM </api/modem.waveform_generator_psk_qam>` waveform
generator features a preamble of multiple reference symbols at the beginning of the communication frame.
They may be removed by specifying the respective property

.. code-block:: python

   operator.waveform_generator.num_preamble_symbols = 0

configuring the number of preamble symbols, resulting in a signal waveform similar to

.. image:: images/getting_started_signal_plot_no_preamble.png
  :alt: PSK/QAM waveform plot without preamble

in which the preamble has clearly been removed.
Describing all configurable parameters is beyond the scope of this introduction,
the API documentation of each processing step should be consulted for detailed descriptions.

A full communication link over an ideal channel model between two dedicated simulated devices
is implemented in the following example:

.. literalinclude:: /../_examples/library/getting_started_link.py
   :language: python
   :linenos:

While at first glance this example can seem somewhat complicated compared to the previous one it is, at its core,
the organic expansion to two devices linked by a single channel model:

* Initially, the required Python modules are imported *(lines 1-5)*
* Handles to the two simulated device entities are created *(lines 7-8)*
* Two identically configured communication modem operators are initialized, operating a device each *(lines 10-16)*
* An ideal channel model linking the two simulated devices is defined *(line 18)*
* An evaluator for the bit errors occurring on the communication link is defined *(line 20)*
* A communication frame is generated and transmitted over the channel model *(lines 22-25)*
* The bit errors are detected and visualized *(lines 27-28)*

This is an example of a core evaluation routine commonly executed in link-level simulations.
However, when considering multiple devices and channel models, as well as performing Monte Carlo style simulations
over scenario parameters, the utilization of the :doc:`Simulation</api/simulation.simulation>` helper class is advised
for optimal scaling.
Its usage is introduced in the next section.

-----------
Simulations
-----------

Evaluating multiple transmissions in scenarios featuring several modems can become quite tedious,
which is why HermesPy offers the :doc:`Simulation </api/simulation.simulation>` helper class.
Considering the same scenario as before, the following snippet demonstrates how
a single communication drop at 40dB signal-to-noise ratio can be generated:

.. literalinclude:: /../_examples/library/getting_started_simulation.py
   :language: python
   :linenos:

Note that lines *5-16* are identical to the previous snippet, defining a scenario with a single receiver
and transmitter modem emitting :doc:`PKS/QAM </api/hermespy.modem.waveform_generator_psk_qam>` waveforms.
However, the scenario is now being managed by the :doc:`Simulation </api/hermespy.simulator_core.simulation>` helper.
It generates and visualizes a single information exchange between all scenario modems *(lines 18)*.
In HermesPy, this is referred to as a :doc:`Drop </api/hermespy.simulator_core.drop>`.
Visualizing the received symbols *(line 19)* and bit errors *(line 20)* during transmission
results in the following constellation and bit error plots:

.. list-table::

    * - .. figure:: /images/getting_started_constellation_low_noise.png

           Symbol Constellation, Low Noise

      - .. figure:: /images/getting_started_errors_low_noise.png

           Bit Errors, Low Noise

Of course, due to the high signal-to-noise ratio and the ideal channel model, no bit errors occur during transmission.
Generating another drop at a much lower ratio, namely 5dB,

.. code-block:: python

   drop = simulation.drop(5.)

leads to several bit-errors during data transmission:

.. list-table::

    * - .. figure:: /images/getting_started_constellation_high_noise.png

           Symbol Constellation, High Noise

      - .. figure:: /images/getting_started_errors_high_noise.png

           Bit Errors, High Noise

.. _GettingStarted_CommandLineTool:

=================
Command Line Tool
=================

This section outlines how to use HermesPy as a command line tool
and provides some reference examples to get new users accustomed with the process of configuring scenarios.

Once HermesPy is installed within any Python environment,
users may call the command line interface by executing the command ``hermes``
in both Linux and Windows command line terminals.
Consult :doc:`/api/hermespy.bin.hermes` for a detailed description of all available command line options.

In short, entering

.. code-block:: bash

   hermes -p /path/to/settings -o /path/to/output

is the most common use-case of the command line interface.
All configuration files located under */path/to/settings* are parsed and interpreted as
an executable scenario configuration.
The configuration is subsequently being executed.
All data resulting from this execution will be stored within */path/to/output*.

If the command-line parameter ``-p`` is left out, then the default path */_settings* will be considered.
If the ``-o`` is left out, then the results will be stored in a unique sub-folder of */results/*.


-----------
First Steps
-----------

Let's start by configuring a basic simulation scenario.
It should consist of:

#. A single transmitting modem, a single receiving modem, both featuring a single ideal antenna
#. A Quadrature-Amplitude modulation scheme
#. A central carrier frequency of 1GHz
#. An ideal channel between both transmitting and receiving modem

This scenario is represented by the following *scenario.yml* file:

.. code-block:: yaml
   :linenos:

   !<Scenario>

   Modems:

      - Transmitter

        carrier_frequency: 1e9

        WaveformPskQam:
            num_data_symbols: 100

      - Receiver

        carrier_frequency: 1e9

        WaveformPskQam:
            num_data_symbols: 100


A second *simulation.yml* file determines the execution behaviour of the scenario.
For now, the baseband-waveforms as well as the mapped symbol constellations
at both the transmitter and receiver should be plotted:

.. code-block:: yaml
   :linenos:

    !<Simulation>

    snr: [20]

    plot_drop: true
    plot_drop_transmitted_signals: true
    plot_drop_received_signals: true
    plot_drop_transmitted_symbols: true
    plot_drop_received_symbols: true

Assuming both *scenario.yaml* and  *simulation.yml* are located within */path/to/settings*, calling

.. code-block:: bash

   hermes -p /path/to/settings

will result in the rendering of four plots displaying the respective information.

A series of configuration files for different waveforms and scenarios is given in ``/_examples/settings``.
For instance, calling

.. code-block:: bash

   hermes -p /settings/chirp_qam

will result in a simulation of a communication system employing QAM-modulated chirps.