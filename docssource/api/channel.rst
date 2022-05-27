=======================
Channel Modeling Module
=======================

The channel modeling module provides functionalities to model
the wireless transmission link between devices on a physical level.

.. autoclasstree:: hermespy.channel
   :strict:
   :namespace: hermespy

The base class of all channel model implementations is defined within

.. toctree::

   channel.channel

The following statistical channel models are currently supported:

.. toctree::

   channel.multipath_fading_channel
   channel.multipath_fading_templates

The following spatial channel models are currently supported:

.. toctree::

   channel.cluster_delay_lines
   channel.cluster_delay_line_indoor_factory
   channel.cluster_delay_line_indoor_office
   channel.cluster_delay_line_rural_macrocells
   channel.cluster_delay_line_street_canyon
   channel.cluster_delay_line_urban_macrocells
   channel.radar_channel
   channel.quadriga