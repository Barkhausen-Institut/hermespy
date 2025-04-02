==========
Beamformer
==========

Beamforming is split into the prototype classes :class:`.TransmitBeamformer` and :class:`.ReceiveBeamformer`
for beamforming operations during signal transmission and reception, respectively.
This is due to the fact that some beamforming algorithms may be exclusive to transmission or reception use-cases.
Should a beamformer be applicable during both transmission and reception both prototypes can be inherited.
An example for such an implementation is the :class:`Conventional <.conventional.ConventionalBeamformer>` beamformer.

.. automodule:: hermespy.beamforming.beamformer
   :private-members: _decode, _encode


.. footbibliography::
