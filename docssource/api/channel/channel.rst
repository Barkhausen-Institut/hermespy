================
Channel Modeling
================

The channel modeling module provides functionalities to model
the wireless transmission link between devices on a physical level.

In general, the transmission of any multi-antenna signal

.. math::

   \mathbf{x}(t) \in \mathbb{C}^{N_\mathrm{Tx}}

considering :math:`N_\mathrm{Tx}` transmitted antenna signal streams
will lead to an excitation of all :math:`N_\mathrm{Rx}` considered reveing antennas 

.. math::

   \mathbf{y}(t) \in \mathbb{C}^{N_\mathrm{Rx}} \ \text{.}

Naturally, this excitation strongly depends on the physical environment and its parameters,
for example the transmitted wavelengths, velocities of transmitter and receiver, distance and 
orientation between transmitter and receiver, environmental scatterers and their respective positions, extents and velocities,
just to name a few.
The environment assumptions, also referred to as channel model, can be described by a linear system

.. math::

   \mathbf{H}(t, \tau, \mathbf{T}_\mathrm{Tx}, \mathbf{T}_\mathrm{Rx}) \in \mathbb{C}^{N_\mathrm{Rx} \times N_\mathrm{Tx}} \ \text{,}

as a function of time :math:`t`, propagation delay :math:`\tau` as well as 
the transmitter's and receiver's respective positions and orientations in space, denoted by the global coordinate system transformation
matrices

.. math::

   \mathbf{T}_\mathrm{Tx} \in \mathbb{R}^{4 \times 4} \ \text{and} \ \mathbf{T}_\mathrm{Rx} \in \mathbb{R}^{4 \times 4} \ \text{,}

respectively.
For ease of notation, the coordinate system transformations will be ommitted in the following discussion, unless specifically required.
The received antenna signals are then the convolution

.. math::

   \mathbf{y}(t) = \int_{-\infty}^{\infty} \mathbf{H}(t, \tau) \mathbf{x}(t - \tau) \,d\tau

of the transmitted signals with the channel model.
Neither :math:`\mathbf{H}(t, \tau)` nor :math:`\mathbf{x}(t)` are generally differentiable.
Therefore HermesPy models the channel and signals as a discrete-time system instead,
sampling the continuous-time models at a rate :math:`f_\mathrm{s}`.
Let

.. math::

   \mathbf{X} &= \left[ \mathbf{x}(0), \mathbf{x}(\frac{1}{f_\mathrm{s}}), \dots, \mathbf{x}(\frac{M_\mathrm{Tx} - 1}{f_\mathrm{s}})  \right] \\
              &= \left[ \mathbf{x}^{(0)}, \mathbf{x}^{(1)}, \dots, \mathbf{x}^{(M_\mathrm{Tx} - 1)} \right] \in \mathbb{C}^{N_\mathrm{Tx} \times M_\mathrm{Tx}}


be the uniformly sampled transmitted signal at sampling rate :math:`f_\mathrm{s}` consisting of :math:`M_\mathrm{Tx}` samples
and :math:`N_\mathrm{Tx}` antenna signal streams.
Equivalently, the received signal can be expressed as

.. math::

   \mathbf{Y} &= \left[ \mathbf{y}(0), \mathbf{y}(\frac{1}{f_\mathrm{s}}), \dots, \mathbf{y}(\frac{M_\mathrm{Rx} - 1}{f_\mathrm{s}})  \right] \\
              &= \left[ \mathbf{y}^{(0)}, \mathbf{y}^{(1)}, \dots, \mathbf{y}^{(M_\mathrm{Rx} - 1)} \right] \in \mathbb{C}^{N_\mathrm{Rx} \times M_\mathrm{Rx}}

the  uniformly sampled received signal at sampling rate :math:`f_\mathrm{s}` consisting of :math:`M_\mathrm{Rx}` samples
and :math:`N_\mathrm{Rx}` antenna signal streams.
Sampling the channel model at the same rate in both time and delay

.. math::

   \mathbf{H}^{(m, \tau)} = \mathbf{H}\left(\frac{m}{f_\mathrm{s}}, \frac{\tau}{f_\mathrm{s}} \right) \ \text{for} \ 
   \substack{
      m = 0 \dots M_\mathrm{Rx} - 1 \\
      \tau = 0 \dots M_\mathrm{Rx} - 1 
   }

enables the expression of the received signal as a time-discrete convolution of the transmitted signal and the channel model

.. math::

   \mathbf{y}^{(m)} = \sum_{\tau = 0}^{m} \mathbf{H}^{(m, \tau)} \mathbf{x}^{(m-\tau)} \ \text{.}

Note that many channel models that consider discrete delay taps :math:`\lbrace \tau_0,\,\dotsc,\,\tau_{\ast} \rbrace`
meaning that the channel is sparse in its delay domain

.. math::

   \mathbf{H}(t, \tau) = \mathbf{0} \ \text{for} \ \tau \notin \lbrace \tau_0,\,\dotsc,\,\tau_\ast \rbrace

and therefore zero for delays outside the set of discrete taps.
Uniformly sampling such models at discrete time-instances results in zero propagation,
since the sampling points will not fall directly on the discrete taps.
To avoid this, HermesPy requires channel models to resample their delay dimension
by either interpolating in between the delays using a sinc-kernel or rounding all delays to the closest sampling tap.  
This behaviour is controlled by the :class:`InterpolationMode<hermespy.channel.channel.InterpolationMode>` flag exposed in several methods.

Each channel model conists of the implementation of a tandem of two abstract base classes:
The :doc:`Channel<channel.Channel>`, which acts as the basic generator,
the :doc:`channel.ChannelRealization`, which represents a specific realization of the channel model,
and the :doc:`channel.ChannelSample`, which represents a specific sample of the channel model in time and space.
The following class diagram visualizes the general interaction:

.. mermaid::

   classDiagram
       
       direction LR 
       
       class Channel {
           <<Abstract>>
           realize() : ChannelRealization
       }

       class ChannelRealization {
           <<Abstract>>
           +sample() : ChannelSample
       }

       class ChannelSample {
           <<Abstract>>
           propagate(Signal) : Signal
       }

       Channel --o ChannelRealization : realize()
       ChannelRealization --o ChannelSample : sample()

       click Channel href "channel.Channel.html"
       click ChannelRealization href "channel.ChannelRealization.html"
       click ChannelSample href "channel.ChannelSample.html"

Each invocation of a :doc:`Channel<channel.Channel>` object's :meth:`realize<hermespy.channel.channel.Channel.realize>` or :meth:`propagate<hermespy.channel.channel.Channel.propagate>` method
results in the generation of a new :doc:`ChannelRealization<channel.ChannelRealization>` object.
A :doc:`ChannelRealization<channel.ChannelRealization>` fixes all spatially invariant parameters of the channel model and realizes the spatially random parameter distributions.
Calling the :doc:`ChannelRealization<channel.ChannelRealization>`'s :meth:`sample<hermespy.channel.channel.ChannelRealization.sample>` method results in a new :doc:`ChannelSample<channel.ChannelSample>` object,
which represents a single one-directional propagation channel in space and time,
meaning calling the :doc:`ChannelSample<channel.ChannelSample>`'s :meth:`propagate<hermespy.channel.channel.ChannelSample.propagate>` method with identical input will result in the identical output.

.. mermaid::

    flowchart LR

    device_alpha_reception[Device Reception]
    device_alpha[Simulated Device]
    device_beta_propagation[Propagated Signal]
    device_alpha_transmission[Device Transmission]

    subgraph Channel
        direction TB
        channel[Channel] --> channel_realization[Channel Realization] --> channel_sample[Channel Sample]
    end

    device_beta_transmission[Device Transmission]
    device_alpha_propagation[Propagated Signal]
    device_beta[Simulated Device]
    device_beta_reception[Device Reception]

    device_alpha_reception --- device_alpha
    device_alpha --> device_alpha_transmission
    device_alpha_transmission --> Channel
    Channel --> device_alpha_propagation --> device_beta
    device_beta --> device_beta_reception
    device_alpha --- device_beta_propagation --- Channel --- device_beta_transmission --- device_beta

    click channel href "channel.Channel.html"
    click channel_realization href "channel.ChannelRealization.html"
    click channel_sample href "channel.ChannelSample.html"
    click device_alpha_propagation href "../core.signal_model.Signal.html"
    click device_beta_propagation href "../core.signal_model.Signal.html"
    click device_alpha href "simulation.simulated_device.html"
    click device_beta href "simulation.simulated_device.html"
    click device_alpha_reception "simulation.simulated_deviceReception.html"
    click device_beta_reception "simulation.simulated_deviceReception.html"
    click device_alpha_transmission "simulation.simulated_deviceTransmission.html"
    click device_beta_transmission "simulation.simulated_deviceTransmission.html"


Operating the described interface requires the import of a :class:`Channel<hermespy.channel.channel.Channel>` implementation and
the :class:`Simulated Devices<hermespy.simulation.simulated_device.SimulatedDevice>` to be linked.

.. literalinclude:: ../../scripts/examples/channel.py
   :language: python
   :linenos:
   :lines: 3-4

Note that, since it is an abstract base class, the :class:`Channel<hermespy.channel.channel.Channel>` used in this example cannot be instantiated directly.
Instead, a concrete implementation such as the :class:`Ideal Channel<hermespy.channel.ideal.IdealChannel>` must be used.
Now, the :class:`Channel<hermespy.channel.channel.Channel>` can be linked to the :class:`Simulated Devices<hermespy.simulation.simulated_device.SimulatedDevice>`.

.. literalinclude:: ../../scripts/examples/channel.py
   :language: python
   :linenos:
   :lines: 13-18

The basic order to simulate a reciprocal channel transmission, meaning both devices
transmit and receive simulataneously, requires the generation of both transmissions,
followed by a propagation over a joint channel realization and finally the reception of both propagations.

.. literalinclude:: ../../scripts/examples/channel.py
   :language: python
   :linenos:
   :lines: 29-40

This snippet is essentially a summary of what is happening within the drop generation of :class:`Simulations<hermespy.simulation.simulation.Simulation>`.
However, by default devices won't generate any waveforms to be transmitted.
For example, a :class:`SimplexLink<hermespy.modem.modem.SimplexLink>` transmitting a :class:`hermespy.modem.waveform.single_carrier.RootRaisedCosine` from the first to the second device can be configured as follows:

.. literalinclude:: ../../scripts/examples/channel.py
   :language: python
   :linenos:
   :lines: 20-27

Investigating the performance of the configured waveform over the specific :class:`Channel<hermespy.channel.channel.Channel>` within a :class:`Simulation<hermespy.simulation.simulation.Simulation>`
requires the instantiation of a new :class:`Simulation<hermespy.simulation.simulation.Simulation>` and adding the already existing :class:`Channel<hermespy.channel.channel.Channel>` and :class:`Simulated Devices<hermespy.simulation.simulated_device.SimulatedDevice>`.

.. literalinclude:: ../../scripts/examples/channel.py
   :language: python
   :linenos:
   :lines: 43-52

Now, evaluators such as :class:`BER<hermespy.modem.evaluators.BitErrorEvaluator>` can be added to the simulation pipeline and
the :class:`Simulation<hermespy.simulation.simulation.Simulation>` can be executed and the results can be analyzed.

.. literalinclude:: ../../scripts/examples/channel.py
   :language: python
   :linenos:
   :lines: 54-59

The channel module consists of the base classes

.. toctree::
   :glob:

   channel.Channel
   channel.ChannelRealization
   channel.ChannelSample

HermesPy provides serveral purely statistical channel models,
which do not assume any spatial correlation between the devices,
their antennas or the propagation environment

.. toctree::
   :glob:
   :maxdepth: 1

   ideal
   fading/fading

In addition, geometry-based stochastical and deterministic channel models are provided,
which model the propagation environment as a collection of scatterers.
Note that these models might require specifying the linked devices :meth:`position<hermespy.simulation.simulated_device.global_position>`.

.. toctree::
   :maxdepth: 1

   cdl/cluster_delay_lines
   delay/delay
   radar/radar
   quadriga

.. toctree::
   :hidden:

   consistent
