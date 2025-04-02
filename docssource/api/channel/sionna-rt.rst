================
SionnaRT Channel
================

.. inheritance-diagram:: hermespy.channel.sionna_rt_channel.SionnaRTChannel hermespy.channel.sionna_rt_channel.SionnaRTChannelRealization hermespy.channel.sionna_rt_channel.SionnaRTChannelSample
   :parts: 1

The :class:`SionnaRTChannel<hermespy.channel.sionna_rt_channel.SionnaRTChannel>` is an adapter for `Sionna Ray Tracing <https://nvlabs.github.io/sionna/api/rt.html>`_ module.

It is deterministic, defined by the given `sionna.rt.Scene <https://nvlabs.github.io/sionna/api/rt.html#scene>`_. Meaning, given same devices and positions, different channel samples (:class:`SionnaRTChannelSample<hermespy.channel.sionna_rt_channel.SionnaRTChannelSample>`) and realizations (:class:`SionnaRTChannelRealization<hermespy.channel.sionna_rt_channel.SionnaRTChannelRealization>`) would not introduce any random changes and would produce equal state and propagation results.

This channel model requires `sionna.rt.Scene` to operate. It should be loaded with `sionna.rt.load_scene` and provided to the :class:`SionnaRTChannel<hermespy.channel.sionna_rt_channel.SionnaRTChannel>` `__init__` method.

Current assumptions in the adapter implementation are:

* All the transmitting and receiving antennas utilize isometric pattern (`"iso"`) and a single vertical polarization (`"V"`).
* The antenna arrays are `synthetic <https://nvlabs.github.io/sionna/api/rt.html?highlight=synthetic_array#sionna.rt.Scene.synthetic_array>`_.
* The delays are not `normalized <https://nvlabs.github.io/sionna/api/rt.html?highlight=normalize_delays#sionna.rt.Paths.normalize_delays>`_.
* If not a single ray hit was cought, then a zeroed signal or state are returned.

It should be noted that `sionna.rt.Scene` is a singleton class. So when a new scene is loaded, a conflict with the previous existing instance will occure. Thus usage of several scenes at once must be avoided.

.. mermaid::

   classDiagram

      direction LR

      class SionnaRTChannel {
   
         _realize() : SionnaRTChannelRealization
      }
   
      class SionnaRTChannelRealization {
   
         _sample() : SionnaRTChannelSample
      }

      class SionnaRTChannelSample {
   
         propagate(Signal) : Signal
      }
   
      SionnaRTChannel --o SionnaRTChannelRealization : realize()
      SionnaRTChannelRealization --o SionnaRTChannelSample : sample()

      click SionnaRTChannel href "#hermespy.channel.sionna_rt_channel.SionnaRTChannel"
      click SionnaRTChannelRealization href "#hermespy.channel.sionna_rt_channel.SionnaRTChannelRealization"
      click SionnaRTChannelSample href "#hermespy.channel.sionna_rt_channel.SionnaRTChannelSample"

The following minimal example outlines how to configure the channel model
within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`:

.. literalinclude:: ../../scripts/examples/channel_SionnaRT.py
   :language: python
   :linenos:
   :lines: 12-36

.. autoclass:: hermespy.channel.sionna_rt_channel.SionnaRTChannel
   :private-members: _realize
   
.. autoclass:: hermespy.channel.sionna_rt_channel.SionnaRTChannelRealization
   :private-members: _sample
   
.. autoclass:: hermespy.channel.sionna_rt_channel.SionnaRTChannelSample

.. footbibliography::
