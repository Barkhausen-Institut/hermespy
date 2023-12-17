===============
Delay Channels
===============

Delay channels offer a simple interface to modeling spatial propagation delays of
signals in a network of distributed :doc:`Devices<simulation.simulated_device.SimulatedDevice>`
and investigate the delay's impact on interference and synchronization.
They, by design, do not model any fading effects but do consider signal attenuation according 
to Frii's transmission formula.

.. mermaid::

   classDiagram
       
      class DelayChannelBase {

         <<Abstract>>

         +realize()
         +propagate()
      }

      class DelayChannelRealization {

         <<Abstract>>

         +propagate()
      }

      class RandomDelayChannel {

         +realize()
         +propagate()
      }

      class RandomDelayChannelRealization {

         +propagate()
      }

      class SpatialDelayChannel {

         +realize()
         +propagate()
      }

      class SpatialDelayChannelRealization {

         +propagate()
      }


      DelayChannelBase o-- DelayChannelRealization : realize()
      RandomDelayChannel o-- RandomDelayChannelRealization : realize()
      SpatialDelayChannel o-- SpatialDelayChannelRealization : realize()
      
      RandomDelayChannel --|> DelayChannelBase
      SpatialDelayChannel --|> DelayChannelBase
      RandomDelayChannelRealization --|> DelayChannelRealization
      SpatialDelayChannelRealization --|> DelayChannelRealization

      click DelayChannelBase href "channel.delay.DelayChannelBase.html"
      click DelayChannelRealization href "channel.delay.DelayChannelRealization.html"
      click RandomDelayChannel href "channel.delay.RandomDelayChannel.html"
      click RandomDelayChannelRealization href "channel.delay.RandomDelayChannelRealization.html"
      click SpatialDelayChannel href "channel.delay.SpatialDelayChannel.html"
      click SpatialDelayChannelRealization href "channel.delay.SpatialDelayChannelRealization.html"

Currently, two types of :doc:`DelayChannels<channel.delay.DelayChannelBase>` are implemented:
:doc:`RandomDelayChannels<channel.delay.RandomDelayChannel>` and :doc:`SpatialDelayChannels<channel.delay.SpatialDelayChannel>`.
Both generate their own type of :doc:`DelayChannelRealization<channel.delay.DelayChannelRealization>`, namely :doc:`RandomDelayChannelRealizations<channel.delay.RandomDelayChannelRealization>`
and :doc:`SpatialDelayChannelRealizations<channel.delay.SpatialDelayChannelRealization>`, respectively.
In general, the delay channel's impulse response between two devices :math:`\alpha` and :math:`\beta` featuring :math:`N^{(\alpha)}` and :math:`N^{(\beta)}` antennas, respectively, is given by

.. math::

   \mathbf{H}(t,\tau) = \frac{1}{4\pi f_\mathrm{c}^{(\alpha)}\overline{\tau}} \mathbf{A}^{(\alpha,\beta)} \delta(\tau - \overline{\tau})

and depends on the assumed propagation delay :math:`\overline{\tau}`, the transmitting device's carrier frequency :math:`f_\mathrm{c}^{(\alpha)}` and the antenna array response :math:`\mathbf{A}^{(\alpha,\beta)}`.
The two implementations differ in the way they generate delay :math:`\overline{\tau}` and antenna array response :math:`\mathbf{A}^{(\alpha,\beta)}`.

.. toctree::
   :glob:
   :hidden:

   channel.delay.*
