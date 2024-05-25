=====================
Spatial Delay Channel
=====================

.. inheritance-diagram:: hermespy.channel.delay.spatial.SpatialDelayChannel hermespy.channel.delay.spatial.SpatialDelayChannelRealization
   :parts: 1

The spatial delay channel requires both linked devices to specify their assumed positions.
Its impulse response between two devices :math:`\alpha` and :math:`\beta` featuring :math:`N^{(\alpha)}` and :math:`N^{(\beta)}` antennas, respectively, is given by

.. math::

    \mathbf{H}(t,\tau) = \frac{1}{4\pi f_\mathrm{c}^{(\alpha)}\overline{\tau}} \mathbf{A}^{(\alpha,\beta)} \delta(\tau - \overline{\tau})\ \text{.}

The assumed propagation delay between the two devices is given by

.. math::

    \overline{\tau} = \frac{\|\mathbf{p}^{(\alpha)} - \mathbf{p}^{(\beta)}\|_2}{c_0}

and depends on the distance between the two devices located at positions :math:`\mathbf{p}^{(\alpha)}` and :math:`\mathbf{p}^{(\beta)}`.
The sensor array response :math:`\mathbf{A}^{(\alpha,\beta)}` depends on the device's relative orientation towards each other.

The following minimal example outlines how to configure the channel model
within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`:

.. literalinclude:: ../../../scripts/examples/channel_SpatialDelayChannel.py
    :language: python
    :linenos:
    :lines: 11-38

.. autoclass:: hermespy.channel.delay.spatial.SpatialDelayChannel
   :private-members: _realize

.. autoclass:: hermespy.channel.delay.spatial.SpatialDelayChannelRealization

.. footbibliography::
