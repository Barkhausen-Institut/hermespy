===============
Ideal Channel
===============

The :doc:`channel.ideal.IdealChannel` is the default :doc:`channel.channel.Channel`
assumed by :doc:`Simulations<simulation.simulation.Simulation>`.
It is completely deterministic and lossless, introducing neither phase shifts, amplitude changes nor propagation delays 
in between the linked :doc:`Devices<core.device.Device>`.

.. mermaid::

   classDiagram

      direction LR

      class IdealChannel {
   
         _realize() : IdealChannelRealization
      }
   
      class IdealChannelRealization {
   
         +propagate(Signal) : ChannelPropagation
      }
   
      IdealChannel --o IdealChannelRealization : realize()
   
      click IdealChannel href "channel.ideal.IdealChannel.html" "IdealChannel"
      click IdealChannelRealization href "channel.ideal.IdealChannelRealization.html" "IdealChannelRealization"

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

.. toctree::
   :glob:
   :hidden:

   channel.ideal.IdealChannel*

.. footbibliography::
