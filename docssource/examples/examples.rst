======================
YAML Examples
======================

This chapter provides several configuration examples to the HermsPy command line interface.
The command line interface allows for the evaluation of wireless scenarios
without in-depth Python knowledge, instead the simulator parameters are configured by
convenient YAML-style files.

In order to run any given example, please follow the :doc:`installation instructions</installation>`
to install the HermesPy suite on your system.
Afterwards, copy and save any given example configuration code to a text-file on your system's drive.
The simulation can now be launched by entering

.. code-block:: bash

   hermes /path/to/downloaded/text-file

in your command line, provided the environment hermes was installed in is activated.
The following example scenarios configurations are currently provided,
giving new users a starting point to define their own scenarios:

.. toctree::
   :maxdepth: 2
   :glob:

   chirp_fsk_lora
   chirp_qam
   hardware_model
   interference_ofdm_single_carrier
   jcas
   ofdm_5g
   ofdm_single_carrier
   operator_separation
   uhd
   audio