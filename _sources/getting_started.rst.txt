Getting Started
===============
This file serves to describe how to make you first steps with Hermespy.

In general, it is recommended to read :doc:`Parameters Description <parameter_description>` at first. In the first section `First Simple Simulation`_, we will describe how you can define a simple simulation. A rather `Complex Simulation`_ is described afterwards.

A few examples with different configurations are given in the **\\_examples** folder

-----------------------
First Simple Simulation
-----------------------

Let's start defining a simple simulation. Let us assume we want to simulate the following:

1. 1 transmitter, 1 receiver modem
2. modulation scheme is PSK/QAM
3. carrier frequency is 1 GHz
4. we have a multipath channel
5. we want to simulate over SNR values between 1 and 30

^^^^^^^^^^^^^
Define Modems
^^^^^^^^^^^^^

Everything concerning **modem setup** needs to be done in the **settings_scenario.ini** file.
Let us open it and create the transmitter modem:

.. code-block:: ini

   [TxModem_1]
   # defines the modulation scheme
   technology_param_file = settings_psk_qam.ini

   # we want to have no encoding
   encoder_param_file = none

   carrier_frequency = 1e9

   tx_power_db = 0

   number_of_antennas = 1
   device_type = BASE_STATION


Let us define the receiver modem as well. Simply insert the following lines below the transmitter modem.

.. code-block:: ini

   [RxModem_1]
   tx_modem = 1

   number_of_antennas = 1

   device_type = UE
   
The parameters are described in :doc:`Parameters Description <parameter_description>`.

Our **first requirement** from our desired simulation is thus fulfilled. Note that the definition of the receiver modem is significantly shorter as technology and coding parameters are chosen from the definition of the respective transmitter modem which is indicated by ``tx_modem = 1``. Carrier frequency is picked from it as well.

Our **second requirement** is to use PSK/QAM as modulation scheme. We did this by setting ``technology_param_file = settings_psk_qam.ini``. **You can change relevant modulation parameters by modifying the **_settings/settings_psk_qam.ini** and by reading :doc:`Parameters Description <parameter_description>` carefully.

The **carrier frequency** was set at the transmitter receiver modem only, as the carrier frequency is taken for the receiver from the respective transmitter modem by setting ``tx_modem = 1``. The carrier frequency at the transmitter side was defined by ``carrier_frequency = 1e9``.

^^^^^^^^^^^^^^^^^^^^^^^^
Define Multipath Channel
^^^^^^^^^^^^^^^^^^^^^^^^

Channels in general are defined between a pair of Modems.

.. note::
   Channels are treated independently in Hermes. It is therefore possible to define multiple different channels at the same time. If you want to define a dependent channel for all the modems, Quadriga needs to be chosen as a parameter.

As we only have one receiver and one transmitter modem, a channel **must** be defined between the of these. For each pair of tx-rx-modem a channel must be defined:

.. code-block:: ini

   [Channel_1_to_1]
   multipath_model = STOCHASTIC
   attenuation_db = 3

   delays = 0, 1e-6, 3e-6
   power_delay_profile_db = 0, -3, -6
   k_rice_db = 3, -inf, -inf

We define a **stochastic** multipath model with an attenuation of 3db. We defined three paths with a delay of ``delays = 0, 1e-6, 3e-6`` per path. The power delay profiles are described in ``power_delay_profile_db`` with respective ``k_rice`` factors. We define one LOS path.

Your **settings_scenario.ini** file should look like this right now:

.. code-block:: ini

   [TxModem_1]
   # defines the modulation scheme
   technology_param_file = settings_psk_qam.ini

   # we want to have no encoding
   encoder_param_file = none

   carrier_frequency = 1e9

   tx_power_db = 0

   number_of_antennas = 1
   device_type = BASE_STATION

   [RxModem_1]
   tx_modem = 1

   number_of_antennas = 1

   device_type = UE
   
   [Channel_1_to_1]
   multipath_model = STOCHASTIC
   attenuation_db = 3

   delays = 0, 1e-6, 3e-6
   power_delay_profile_db = 0, -3, -6
   k_rice_db = 3, -inf, -inf

Let us fulfill the **fifth requirement** right now.

^^^^^^^^^^^^^^^^^^
 Tweak Simulation
^^^^^^^^^^^^^^^^^^

Simatulion related parameters are to be changed in **settings_general.ini**.

This file is full of default values, which are related to the simulation. To keep it simple, let us only change the SNR values to loop over right now. As this concerns the **NoiseLoop**, the respective SNR-vector can be found there:

.. code-block:: ini

   snr_vector = np.arange(0, 30, 1)

Save it and then run the simulation!

------------------
Complex Simulation
------------------

Le us define a more complex simulation now. We want to have

1. Three Transmitters and two receivers. The transmitters send at 1GHz, 1.5Ghz, and 2Ghz.
2. Modulation Schemes should be Chirp-FSK, PSK/QAM, and OFDM.
3. The two receivers should listen to PSK/QAM and OFDM.
4. The OFDM Modem should use LDPC Encoding.
5. SNR type should be ``Es/NO(dB)``.
6. We want to have one 5G Phy channel model for the OFDM pair and a COST-259 Model for the other pair.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Define Transmitters and Receivers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**settings_scenario.ini** is once again your friend. Let us define the **three transmitters**:

.. code-block:: ini

   [TxModem_1]
   technology_param_file = settings_chirp_fsk.ini
   encoder_param_file = none

   carrier_frequency = 1e9

   tx_power_db = 0

   number_of_antennas = 1

   device_type = BASE_STATION

   [TxModem_2]
   technology_param_file = settings_psk_qam.ini
   encoder_param_file = none

   carrier_frequency = 1.5e9

   tx_power_db = 0

   number_of_antennas = 1

   device_type = BASE_STATION

   [TxModem_3]
   technology_param_file = settings_ofdm.ini
   encoder_param_file = settings_ldpc_encoder.ini

   carrier_frequency = 2e9

   tx_power_db = 0

   number_of_antennas = 2

   device_type = BASE_STATION

``[TxModem_<i>]`` sections, ``i`` being the 1-index based transmitter modem indices, denote the transmitter definitions. We have three as required by our **first bulletpoint**. ``carrier_frequency`` are set as required.

The **second requirement** is fulfilled by setting the ``technology`` parameter. Note that we changed the encoding of ``[TxModem_3]`` (our OFDM modem) to ``encoding_param_file = settings_ldpc_encoder.ini``. Therefore, we use the LDPC encoder defined by the **_settings/coding/settings_ldpc_encoder.ini** file! Maybe you realized that we changed the ``number_of_antennas`` to 2 for our OFDM modem.

Let's define the **receiver modems**. They are quite easy as the important parts are already defined by the transmitters:

.. code-block:: ini

   [RxModem_1]
   tx_modem = 2

   number_of_antennas = 1

   device_type = UE

   [RxModem_2]
   tx_modem = 3

   number_of_antennas = 2

   device_type = UE

We want to have two receivers, as opposed to three transmitters. Therefore, we have only the sections ``[RxModem_1]`` and ``[RxModem_2]``. The receiver modems needs to know to which transmitter they are "connected", therefore ``tx_modem`` needs to be set accordingly. The technology, carrier frequency, and coding are set internally in accordance to this pairing. In our case, ``[RxModem_1]`` listens to ``[TxModem_2]`` and ``[RxModem_2]`` listens to ``[TxModem_3]``. For our OFDM modem, we also defined the number of antennas. Thus, **requirements 1 to 4 are fulfilled right now**.

.. note::

   We also changed the ``device_type``. This is important for the channel definition later on.

Let's continue with the simulation.

^^^^^^^^^^
Simulation
^^^^^^^^^^

The default values are quite fine for simulation purposes usually. However, we want to change the SNR type. Let's do it:

.. code-block:: ini

   [NoiseLoop]
   snr_type = Es/N0(dB)

We successfully changed the ``snr_type``.

^^^^^^^^^^^^^^^^^^
Channel Definition
^^^^^^^^^^^^^^^^^^

Although no receiver is listening to ``[TxModem_1]``, there might be interferences occurring if the carrier frequencies are close to each other. Therefore channels need to be defined to this modem as well. In general:

.. note::

   There must be a channel definition between each possible tx-rx-pair, independent of the fact if the very tx-rx-pair listens to each other or not. If we have ``N_T`` transmitters and ``N_R`` receivers, ``N_T * N_R`` channels need to be defined.

For simplicity's sake, let's say that all tx-rx-channel have an AWGN channel, i.e.:

.. code-block:: ini

   [Channel_<i>_to_<j>]
   multipath_model = NONE

``i`` denotes the **transmitter** and ``j`` denotes the **receiver**. However, we have two exceptions as defined in **bulletpoint 6**:

6. We want to have one 5G Phy channel model for the OFDM pair and a COST-259 Model for the other pair.

OFDM-pair means ``i=3``, ``j=2``, i.e:

.. code-block:: ini

   [Channel_3_to_2]
   multipath_model = 5G_TDL
   tdl_type = A
   rms_delay = 90
   correlation = LOW
   custom_correlation = 0.5

Please check the description in :doc:`Parameter Description <parameter_description>` for a detailed description of the parameters.

For the PSK/QAM pair, we want to have a COST-259 Channel model. In this case, ``i=2, j=1``, yielding:

.. code-block:: ini

   [Channel_2_to_1]
   multipath_model = COST259
   cost_type = hilly_terrain

That should be self explanatory.

In total our **settings_scenario.ini** - file should look like this:

.. code-block:: ini

   [TxModem_1]
   technology_param_file = settings_chirp_fsk.ini
   encoder_param_file = none

   carrier_frequency = 1e9

   tx_power_db = 0

   number_of_antennas = 1

   device_type = BASE_STATION

   [TxModem_2]
   technology_param_file = settings_psk_qam.ini
   encoder_param_file = none

   carrier_frequency = 1.5e9

   tx_power_db = 0

   number_of_antennas = 1

   device_type = BASE_STATION

   [TxModem_3]
   technology_param_file = settings_ofdm.ini
   encoder_param_file = settings_ldpc_encoder.ini

   carrier_frequency = 2e9

   tx_power_db = 0

   number_of_antennas = 2

   device_type = BASE_STATION

   [RxModem_1]
   tx_modem = 2

   number_of_antennas = 1

   device_type = UE

   [RxModem_2]
   tx_modem = 3

   number_of_antennas = 2

   device_type = UE

   [Channel_1_to_1]
   multipath_model = NONE

   [Channel_2_to_1]
   multipath_model = COST259
   cost_type = hilly_terrain

   [Channel_3_to_1]
   multipath_model = NONE

   [Channel_1_to_2]
   multipath_model = NONE

   [Channel_2_to_2]
   multipath_model = NONE

   [Channel_3_to_2]
   multipath_model = 5G_TDL
   tdl_type = A
   rms_delay = 90
   correlation = LOW
   custom_correlation = 0.5

.. note::

   The order the Channels are defined is of no importance. That means, it does not matter if you start by defining ``[Channel_1_to_1]`` or ``[Channel_2_to_2]`` for the first channel. It is only important that you define all possible channels.
