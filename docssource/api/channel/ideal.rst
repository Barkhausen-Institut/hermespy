===============
Ideal Channel
===============


.. inheritance-diagram:: hermespy.channel.ideal.IdealChannel hermespy.channel.ideal.IdealChannelRealization hermespy.channel.ideal.IdealChannelSample
   :parts: 1


The :class:`IdealChannel<hermespy.channel.ideal.IdealChannel>` is the default :class:`Channel<hermespy.channel.channel.Channel>`
assumed by :class:`Simulations<hermespy.simulation.simulation.Simulation>`.
It is completely deterministic and lossless, introducing neither phase shifts, amplitude changes nor propagation delays 
in between the linked :class:`Devices<hermespy.simulation.simulated_device.SimulatedDevice>`.

.. mermaid::

   classDiagram

      direction LR

      class IdealChannel {
   
         _realize() : IdealChannelRealization
      }
   
      class IdealChannelRealization {
   
         _sample() : IdealChannelSample
      }

      class IdealChannelSample {
   
         propagate(Signal) : Signal
      }
   
      IdealChannel --o IdealChannelRealization : realize()
      IdealChannelRealization --o IdealChannelSample : sample()

      click IdealChannel href "#hermespy.channel.ideal.IdealChannel"
      click IdealChannelRealization href "#hermespy.channel.ideal.IdealChannelRealization"
      click IdealChannelSample href "#hermespy.channel.ideal.IdealChannelSample"

Considering two devices :math:`\alpha` and :math:`\beta` featuring :math:`N_\alpha` and :math:`N_\beta` antennas respectively,
the ideal Channel's impulse response

.. math::

   \mathbf{H}(t, \tau) = \delta(\tau) 
   \left\lbrace\begin{array}{cr}
      \left[1, 1,\,\dotsc,\, 1 \right] & \text{for } N_\beta = 1 \\
      \left[1, 1,\,\dotsc,\, 1 \right]^\mathsf{T} & \text{for } N_\alpha = 1 \\
      \begin{bmatrix}
         1, & 0, & \dots, & 0 \\
         0, & 1, & \dots, & 0 \\
         \vdots & \vdots & \ddots & \vdots \\
         0, & 0, & \dots, & 1
      \end{bmatrix} & \text{otherwise}
   \end{array}\right\rbrace \in \mathbb{C}^{N_\beta \times N_\alpha}

depends on the number of antennas of the devices and is independent of the time :math:`t`.
For channels with an unequal number of antennas, the ideal Channel's impulse response is a diagonal matrix with ones on the diagonal,
padded with zeros to match the dimensions of the channel matrix.
Therefore, the device with the bigger amount of antennas will receive / transmit nothing
from the additional antennas.

.. autoclass:: hermespy.channel.ideal.IdealChannel
   :private-members: _realize

.. autoclass:: hermespy.channel.ideal.IdealChannelRealization
   :private-members: _sample

.. autoclass:: hermespy.channel.ideal.IdealChannelSample
   :private-members: _propagate

.. footbibliography::
