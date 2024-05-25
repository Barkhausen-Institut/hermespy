=============
Indoor Office
=============

.. inheritance-diagram:: hermespy.channel.cdl.indoor_office.IndoorOffice hermespy.channel.cdl.indoor_office.IndoorOfficeRealization
   :parts: 1

Implementation of an indoor office communication channel model.
Refer to the :footcite:t:`3GPP:TR38901` for detailed information.

The following minimal example outlines how to configure the channel model
within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`:

.. literalinclude:: ../../../scripts/examples/channel_cdl_indoor_office.py
   :language: python
   :linenos:
   :lines: 12-40
    
.. autoclass:: hermespy.channel.cdl.indoor_office.IndoorOffice

.. autoclass:: hermespy.channel.cdl.indoor_office.IndoorOfficeRealization

.. autoclass:: hermespy.channel.cdl.indoor_office.OfficeType

.. footbibliography::
