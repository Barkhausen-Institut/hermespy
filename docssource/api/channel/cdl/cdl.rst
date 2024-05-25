==========
Static CDL
==========

.. inheritance-diagram:: hermespy.channel.cdl.cdl.CDL hermespy.channel.cdl.cdl.CDLRealization
   :parts: 1

.. mermaid::

   classDiagram

      direction LR

      class CDL {
   
         _realize() : CDLRealization
      }
   
      class CDLRealization {
   
         _sample() : CDLSample
      }

      class ClusterDelayLineSample {
   
         propagate(Signal) : Signal
      }
   
      CDL --o CDLRealization : realize()
      CDLRealization --o ClusterDelayLineSample : sample()

      click CDL href "#hermespy.channel.cdl.cdl.CDL"
      click CDLRealization href "#hermespy.channel.cdl.cdl.CDLRealization"
      click ClusterDelayLineSample href "cluster_delay_lines.html#hermespy.channel.cdl.cluster_delay_lines.ClusterDelayLineSample"

.. autoclass:: hermespy.channel.cdl.cdl.CDL
   :private-members: _realize

.. autoclass:: hermespy.channel.cdl.cdl.CDLRealization
   :private-members: _sample

.. autoclass:: hermespy.channel.cdl.CDLType

.. footbibliography::
