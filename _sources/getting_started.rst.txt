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
* :doc:`Channel Codings</api/fec.coding>` as the channel coding configuration
* :doc:`Waveform Generators</api/modem.waveform>` as the transmit waveform configuration
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
  :doc:`Waveform Generator </api/modem.waveform>` instance to the modem's
  waveform_generator property *(line 7)*. |br|
  In our case, this is an instance of a :doc:`Single Carrier </api/modem.waveform_single_carrier>`
  waveform.
* The device on which the modem operates is defined *(line 9)*.
* An signal model, encoding a single communication frame, emitted by the modem operator
  is generated and plotted *(lines 10-13)*

Executing the snippet will result in a plot similar to

.. figure:: images/getting_started_signal_plot.png
  :align: center
  :alt: PSK/QAM default waveform plot
  :scale: 75%

  Signal Model Plot

which visualizes the generated samples in time-domain (left sub-plot) and its respective
discrete fourier transform (right sub-plot).

While this is only a minimal example, it highlights the philosophy behind the HermesPy API, namely that
each signal processing step is represented by a class modeling its functionality.
Instances of those classes are assigned to property slots, where they will be executed sequentially
during signal generation.
Changing the waveform generated by the modem operator defined in the previous snippet
is therefore as simple as assigning a different type of
:doc:`Waveform Generator </api/modem.waveform>`
to its :meth:`hermespy.modem.modem.Modem.waveform_generator` property slot.

Of course, a multitude of parameters can be configured to customize the behaviour of each processing step.
For instance, the frame generated by a :doc:`Single Carrier </api/modem.waveform_single_carrier>` waveform
generator features no preamble by default.
A preamble is defined as a static set of known reference symbols at the beginning of the communication frame.
By modifying the property

.. code-block:: python

   operator.waveform_generator.num_preamble_symbols = 20


the user may freely chose the number of preamble symbols.
In this case, requesting :math:`20` symbols results in a generated frame

.. figure:: images/getting_started_signal_plot_preamble.png
  :alt: PSK/QAM waveform plot with preamble
  :align: center
  :scale: 75%

  Signal Model Plot with Preamble

featuring the added preamble.
Describing all configurable parameters is beyond the scope of this introduction,
the :doc:`API <api/api>` documentation of each processing step should be consulted for detailed descriptions.
In general, each settable property may be freely configured by the user.

While the previous code snippet highlighted how to generate basic waveform models,
link-level simulations usually consider the signal exchange between two dedicated devices.
A full communication link over an ideal channel model between two dedicated simulated devices
is implemented in the next example:

.. literalinclude:: /../_examples/library/getting_started_link.py
   :language: python
   :linenos:

While this code may seem somewhat complex at first glance, it expands the previous example by some important
concepts, namely :class:`Channels <hermespy.channel.channel.Channel>`
and :class:`Evaluators <hermespy.core.monte_carlo.Evaluator>`.
:class:`Channels <hermespy.channel.channel.Channel>` are the key simulation entity modeling
waveform propagation between devices.
Depending on the simulation assumptions, users may select from a multitude of different classes providing specific
model implementations.
Refer to the :doc:`Channel Module<api/channel>` for a detailed overview.
:class:`Evaluators <hermespy.core.monte_carlo.Evaluator>` HermesPy's abstraction for the extraction of specific
performance indicators from simulation objects.
In theory, almost any object and its respective properties can be used to implement custom evaluation routines.
For communication evaluations, several default evaluation routines are already shipped within the
:doc:`Communication Evaluators<api/modem.evaluators>` module.

Executing the snippet above results in two visualizations being rendered after propagation simulation,

.. list-table::

    * - .. figure:: /images/getting_started_constellation_low_noise.png
           :align: center
           :alt: Constellation

           Symbol Constellation, Noiseless

      - .. figure:: /images/getting_started_errors_low_noise.png
           :align: center
           :alt: BER

           Bit Errors, Noiseless

namely a symbol constellation diagram at the receiver side and a bit error evaluation stem graph.
Since the channel we model is actually an ideal channel and no noise is added at the receiver,
no bit errors occur during data transmission.
Adapting line 28 of the snippet according to

.. code-block:: python

   rx_device.receive(rx_signal, snr=4.)

will result in additive white gaussian being added at the receiver side with a signal to noise ratio of :math:`4`.
As a consequence, the constellation gets distorted, leading to false decisions during demodulation and therefore
to a number of bit errors during data transmission.
Executing the snippet with noise consideration results in a visualization similar to

.. list-table::

    * - .. figure:: /images/getting_started_constellation_high_noise.png
           :align: center
           :alt: Constellation

           Symbol Constellation, Noisy

      - .. figure:: /images/getting_started_errors_high_noise.png
           :align: center
           :alt: BER

           Bit Errors, Noisy

where said effects are clearly visible.

Propagating signal models over a channel linking two devices is an example of one of the fundamental
routines commonly executed in link-level simulations.
However, complex investigations usually consider multiple devices and channel models,
as well as perform Monte Carlo style simulations over a grid of model parametrizations,
which can lead to computationally complex routines, even for seemingly simple scenarios.
In order to streamline simulation definition and execution, HermesPy provides the
:doc:`Simulation</api/simulation.simulation>` helper class, which automizes the process
of distributing the simulation workload in multicore systems and parameter grid evaluations.
Its usage is introduced in the next section.

-----------
Simulations
-----------

Consider the simulation scenario of a single device transmitting its waveforms and receiving them back
after reflections from surroundings, assuming ideal isolation between transmit and receive chain.
One of the most frequently conducted investigations in communication signal processing is the estimation of the
bit error rate (BER) in relation to the noise power at the receiver side of the communication link.
Within HermesPy, :doc:`Simulations</api/simulation.simulation>` can be configured to estimate performance indicators
such as bit error rate over arbitrary parameter dimensions.
For example, the following snippet

.. literalinclude:: /../_examples/library/getting_started_simulation.py
   :language: python
   :linenos:

defines the described scenario, adds a bit error rate evaluation and, most importantly,
defines a sweep over the (linear) signal to noise ratio from :math:`10` to :math:`5`,
collecting :math:`1000` samples for each sweep point, respectively.
Executing the script will launch a full simulation run and a rendered result

.. figure:: images/getting_started_ber_evaluation.png
   :alt: Bit Error Rate Plot
   :align: center
   :scale: 75%

   Bit Error Rate Evaluation

of the bit error rate graph.

Now, a typical approach to reduce the bit errors is the introduction of :doc:`Channel Coding<api/fec>`
schemes for error correction.
They introduce redundancy within the transmitted bit stream during transmission
and exploit said redundancy at the receiver to correct errors.
One of the most basic error-correcting channel codes is the :doc:`Repetition Encoder<api/fec.repetition>`,
which simply repeats bits to be transmitted and decodes by majority voting after reception.
In theory, the more repetitions per transmitted data frame, the higher the error correction capabilites.
But the more redundancy is introduced, the lower the actual information throughput becomes.
Therefore, there is a sweet-spot within the tradeoff between data throughput and repetitions for a given signal to noise
ratio.

The following snippet configures HermesPy to conduct a simulation visualizing the data rate relative
to number of repetitions and noise ratio:

.. literalinclude:: /../_examples/library/getting_started_simulation_multidim.py
   :language: python
   :linenos:

Executing it leads to the rendering of a surface plot visualization, from which engineers
can infer the selection of a proper repetition rate in order to achieve a required data rate for
a given noise ratio:

.. figure:: /images/getting_started_simulation_multidim_drx.png
   :align: center
   :alt: Throughput
   :scale: 75%

   Data Throughput

.. _GettingStarted_CommandLineTool:

=================
Command Line Tool
=================

This section outlines how to use HermesPy as a command line tool
and provides some reference examples to get new users accustomed with the process of configuring scenarios.

Once HermesPy is installed within any Python environment,
users may call the command line interface by executing the command :mod:`hermes <hermespy.bin.hermes>`
in both Linux and Windows command line terminals.
Consult :doc:`/api/bin.hermes` for a detailed description of all available command line options.

In short, entering

.. code-block:: bash

   hermes /path/to/config.yml -o /path/to/output

is the most common use-case of the command line interface.
The configuration */path/to/config.yml* is subsequently being executed.
All data resulting from this execution will be stored within */path/to/output*.

If the ``-o`` is left out, then the results will be stored in a unique sub-folder of */results/*.


-----------
First Steps
-----------

Let's begin by configuring a basic simulation scenario.
It should consist of:

#. A single device featuring a single omnidirectional antenna
#. A modem operator with

   * :math:`R = \frac{1}{3}` repetition coding
   * 100GHz symbol rate
   * Root-Raised-Cosine waveforms
   * a modulation order of 16
   * a frame consisting of 10 preamble- and 1000 data symbols

#. An evaluation routine for the bit error rate
#. A parameter sweep over the SNR between 0 and 20 dB

This scenario is represented by the following *simulation.yml* file:

.. literalinclude:: /../_examples/settings/chirp_qam.yml
   :language: yaml
   :linenos:

Assuming *simulation.yml* is located within */path/to/settings*, calling

.. code-block:: bash

   hermes /path/to/config.yml

will result in the rendering of four plots displaying the respective information.
The resulting plots and a matlab dump of the evaluation data will be saved in your current working directory.
For different types of configurations, please refer to the `Examples`_ folder within the HermesPy
Github repository.

.. _examples: https://github.com/Barkhausen-Institut/hermespy/tree/main/_examples/settings