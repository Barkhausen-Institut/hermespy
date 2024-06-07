========
Channel
========

.. inheritance-diagram:: hermespy.channel.channel.Channel
   :parts: 1


The channel model represents the basic configuration of two linked :class:`SimulatedDevices<hermespy.simulation.simulated_device.SimulatedDevice>`
:meth:`alpha_device<.alpha_device>` and :meth:`beta_device<.beta_device>` exchanging electromagnetic :class:`Signals<hermespy.core.signal_model.Signal>`.

Each invokation of :meth:`realize<hermespy.channel.channel.Channel.realize>` will generate a new :class:`ChannelRealization<hermespy.channel.channel.ChannelRealization>` instance by internally calling :meth:`._realize<hermespy.channel.channel.Channel._realize>`.
The channel model represents the matrix function of time :math:`t` and delay :math:`\tau`

.. math::

   \mathbf{H}(t, \tau; \mathbf{\zeta}) \in \mathbb{C}^{N_{\mathrm{Rx}} \times N_{\mathrm{Tx}}} \ \text{,}

the dimensionality of which depends on the number of transmitting antennas :math:`N_{\mathrm{Tx}}` and number of receiving antennas :math:`N_{\mathrm{Rx}}`.
The vector :math:`\mathbf{\zeta}` represents the channel model's paramteres as random variables.
Realizing the channel model is synonymous with realizing and "fixing" these random parameters by drawing a sample from their respective
distributions, so that a :class:`ChannelRealization<hermespy.channel.channel.ChannelRealization>` represents the deterministic function

.. math::

   \mathbf{H}(t, \tau) \in \mathbb{C}^{N_{\mathrm{Rx}} \times N_{\mathrm{Tx}}} \ \text{.}

.. autoclass:: hermespy.channel.channel.Channel

.. autoclass:: hermespy.channel.channel.ChannelSampleHook

.. autoclass:: hermespy.channel.channel.InterpolationMode

.. footbibliography::
