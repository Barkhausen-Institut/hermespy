====================
Channel Estimation
====================

.. inheritance-diagram:: hermespy.modem.waveform.ChannelEstimation
   :parts: 1

Channel estimation is the process of estimating the channel state :math:`\widehat{\mathbf{H}}` during signal reception by observing the channel's effect on known reference symbols distributed over the communication frame.
Within the Hermes communication processing pipeline, channel estimation is directly following the demodulation stage.
Several other processing steps rely on the channel estimation, namely symbol and stream coding during both transmission and reception, as well as the final equalization step before unmapping.

During simulation runtime the ideal channel state information used to propagate the transmitted and received signals is available, so that users may configure an ideal channel state estimator leading to :math:`\widehat{\mathbf{H}} = \mathbf{H}`.
When using Hermes in hardware loop mode operating SDRs or when configuring non-ideal channel state estimators during simulations, transmit processing blocks only have access to the channel state estimated from the most recent reception.

.. autoclass:: hermespy.modem.waveform.ChannelEstimation

.. footbibliography::
