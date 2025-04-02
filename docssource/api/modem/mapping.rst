=============
Mapping
=============

.. inheritance-diagram:: hermespy.modem.tools.psk_qam_mapping.PskQamMapping
   :parts: 1

Mapping between bits and PSK/QAM/PAM constellation
This module provides a class for a PSK/QAM mapper/demapper.

The following features are supported:

* arbitrary 2D (complex) constellation mapping can be given
* default Gray-coded constellations for BPSK, QPSK, 8-PSK, 4-, 8-, 16- PAM, 16-, 64- and 256-QAM are provided
* all default constellations follow 3GPP standards TS 36.211 (except 8-PSK, which is not defined in 3GPP)
* hard and soft (LLR) output are available

This implementation has currently the following limitations:

* LLR available only for default BPSK, QPSK, 4-, 8-, 16- PAM, 16-, 64- and 256-QAM
* only linear approximation of LLR is considered, similar to the one described in:

Tosato, Bisaglia, "Simplified Soft-Output Demapper for Binary Interleaved COFDM with
Application to HIPERLAN/2", Proceedings of IEEE International Commun. Conf. (ICC) 2002

.. autoclass:: hermespy.modem.tools.psk_qam_mapping.PskQamMapping

.. footbibliography::