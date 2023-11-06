========
Quadriga
========

The QUAsi Determinstic RadIo channel GenerAtor (QuaDRiGa) channel model
is a Matlab based tool for generating radio channel impulse responses
in multi-node networks of radio devices :footcite:`2014:jaeckel,2014:burkhardt`.
It is integrated into Hermes as a channel model plugin, with the generation of new channel realizations
triggering an execution of Quadriga's Matlab scripts via either the Matlab Python API or Octave.
Please refer to the :doc:`/installation` hints for the required setup.

.. mermaid::
   :align: center

   classDiagram
       
      class QuadrigaChannel {

         +realize()
         +propagate()
      }

      class QuadrigaChannelRealization {

         +propagate()
      }

      class QuadrigaInterface {

         <<Abstract>>

         +get_impulse_response()
      }

      QuadrigaChannel o-- QuadrigaChannelRealization : realize()
      
      QuadrigaChannel *-- QuadrigaInterface

      click QuadrigaChannel href "channel.quadriga.QuadrigaChannel.html"
      click QuadrigaChannelRealization href "channel.quadriga.QuadrigaChannelRealization.html"
      click QuadrigaInterface href "channel.quadriga.QuadrigaInterface.html"

.. toctree::
   :glob:
   :hidden:

   channel.quadriga.*

.. footbibliography::
