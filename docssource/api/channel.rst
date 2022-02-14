=======================
Channel Modeling Module
=======================

The channel modeling module provides functionalities to model
the wireless transmission link between devices on a physical level.

.. autoclasstree:: hermespy.channel
   :strict:
   :namespace: hermespy.channel

The base class of all channel model implementations is defined within

.. toctree::

   channel.channel

The following statistical / spatial channel models are currently supported:

.. toctree::

   channel.multipath_fading_channel
   channel.multipath_fading_templates
   channel.radar_channel
   channel.quadriga