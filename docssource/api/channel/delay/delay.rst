===============
Delay Channels
===============

.. inheritance-diagram:: hermespy.channel.delay.delay.DelayChannelBase hermespy.channel.delay.delay.DelayChannelRealization hermespy.channel.delay.delay.DelayChannelSample
   :parts: 1

Delay channels offer a simple interface to modeling spatial propagation delays of
signals in a network of distributed :class:`Devices<hermespy.simulation.simulated_device.SimulatedDevice>`
and investigate the delay's impact on interference and synchronization.
They, by design, do not model any fading effects but do consider signal attenuation according 
to Frii's transmission formula.

.. mermaid::

   classDiagram
       
      direction LR 

      class DelayChannelBase {
         <<Abstract>>
         +realize() : DelayChannelRealization
      }

      class DelayChannelRealization {
         <<Abstract>>
         +sample() : DelayChannelSample
      }

      class DelayChannelSample {
         +propagate(Signal) : Signal
      }

      DelayChannelBase --> DelayChannelRealization : realize()
      DelayChannelRealization --> DelayChannelSample : sample()

      click DelayChannelBase href "#hermespy.channel.delay.DelayChannelBase"
      click DelayChannelRealization href "#hermespy.channel.delay.DelayChannelRealization"
      click DelayChannelSample href "#hermespy.channel.delay.DelayChannelSample"

Currently, two types of :class:`DelayChannels<hermespy.channel.delay.delay.DelayChannelBase>` are implemented:
:class:`RandomDelayChannels<hermespy.channel.delay.random.RandomDelayChannel>` and :class:`SpatialDelayChannels<hermespy.channel.delay.spatial.SpatialDelayChannel>`.
Both generate their own type of :class:`DelayChannelRealization<hermespy.channel.delay.delay.DelayChannelRealization>`, namely :class:`RandomDelayChannelRealizations<hermespy.channel.delay.random.RandomDelayChannelRealization>`
and :class:`SpatialDelayChannelRealizations<hermespy.channel.delay.spatial.SpatialDelayChannelRealization>`, respectively.
In general, the delay channel's impulse response between two devices :math:`\alpha` and :math:`\beta` featuring :math:`N^{(\alpha)}` and :math:`N^{(\beta)}` antennas, respectively, is given by

.. math::

   \mathbf{H}(t,\tau) = \frac{1}{4\pi f_\mathrm{c}^{(\alpha)}\overline{\tau}} \mathbf{A}^{(\alpha,\beta)} \delta(\tau - \overline{\tau})

and depends on the assumed propagation delay :math:`\overline{\tau}`, the transmitting device's carrier frequency :math:`f_\mathrm{c}^{(\alpha)}` and the antenna array response :math:`\mathbf{A}^{(\alpha,\beta)}`.
The two implementations differ in the way they generate delay :math:`\overline{\tau}` and antenna array response :math:`\mathbf{A}^{(\alpha,\beta)}`.

.. toctree::
   :hidden:

   spatial
   random

.. autoclass:: hermespy.channel.delay.delay.DelayChannelBase
   :private-members: _realize

.. autoclass:: hermespy.channel.delay.delay.DelayChannelRealization
   :private-members: _sample

.. autoclass:: hermespy.channel.delay.delay.DelayChannelSample

.. autoclass:: hermespy.channel.delay.delay.DCRT

.. footbibliography::
