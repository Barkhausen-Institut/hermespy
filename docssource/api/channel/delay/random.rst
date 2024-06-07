=====================
Random Delay Channel
=====================

.. inheritance-diagram:: hermespy.channel.delay.random.RandomDelayChannel hermespy.channel.delay.random.RandomDelayChannelRealization

Delay channel assuming a uniformly distributed random propagation between the linked devices.
Its impulse response between two devices :math:`\alpha` and :math:`\beta` featuring :math:`N^{(\alpha)}` and :math:`N^{(\beta)}` antennas, respectively, is given by

.. math::

    \mathbf{H}(t,\tau) = \frac{1}{4\pi f_\mathrm{c}^{(\alpha)}\overline{\tau}} \mathbf{A}^{(\alpha,\beta)} \delta(\tau - \overline{\tau})\ \text{.}

The assumed propagation delay is drawn from the uniform distribution

.. math::

    \overline{\tau} \sim \mathcal{U}(\tau_{\mathrm{Min}}, \tau_{\mathrm{Max}})

and lies in the interval between :math:`\tau_\mathrm{Min}` and :math:`\tau_\mathrm{Max}`.
The sensor array response :math:`\mathbf{A}^{(\alpha,\beta)}` is always assumed to be the identity matrix.

The following minimal example outlines how to configure the channel model
within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`:

.. literalinclude:: ../../../scripts/examples/channel_RandomDelayChannel.py
    :language: python
    :linenos:
    :lines: 11-38

.. autoclass:: hermespy.channel.delay.random.RandomDelayChannel
   :private-members: _realize

.. autoclass:: hermespy.channel.delay.random.RandomDelayChannelRealization

.. footbibliography::
