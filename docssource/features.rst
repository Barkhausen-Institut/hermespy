********
Features
********

The feature set of HermesPy is steadily expanding.
This list provides an overview of core functionalities, latest feature additions introduced with the last official release are denoted in **bold**.
Full releases with a new set of features will be provided on a bi-anual basis in Spring and Fall, with software patches in between.


.. list-table:: **Beamforming**
   :header-rows: 1

   * - Feature
     - API References
     - Examples

   * - Capon Beamformer
     - :doc:`api/beamforming.capon`
     - :doc:`notebooks/beamformer`

   * - Conventional Beamformer
     - :doc:`api/beamforming.conventional`
     - :doc:`notebooks/beamformer`

.. list-table:: **Forward Error Correction**
   :header-rows: 1

   * - Feature
     - API References
     - Examples

   * - Interleaving
     - :doc:`api/fec.block_interleaver`
     -

   * - Cyclic Reundancy Check Mock
     - :doc:`api/fec.crc`
     -

   * - Low Differential Parity Checks
     - :doc:`api/fec.ldpc`
     - :doc:`examples/ofdm_5g`

   * - Polar Codes
     - :doc:`api/fec.polar`
     -

   * - Repetition Coding
     - :doc:`api/fec.repetition`
     - :doc:`examples/chirp_fsk_lora`, :doc:`getting_started`

   * - Reed Solomon Codes
     - :doc:`api/fec.rs`
     -

   * - Recursive Systematic Convolutional Codes
     - :doc:`api/fec.rsc`
     - 

   * - Scrambling
     - :doc:`api/fec.scrambler`
     -

   * - Turbo Coding
     - :doc:`api/fec.turbo`
     - 

.. list-table:: **Channel Models**
   :header-rows: 1

   * - Model
     - API References
     - Examples

   * - Indoor Factory
     - :doc:`api/channel.cluster_delay_line_indoor_factory`
     -

   * - Indoor Office
     - :doc:`api/channel.cluster_delay_line_indoor_office`
     -

   * - Rural Macrocells
     - :doc:`api/channel.cluster_delay_line_rural_macrocells`
     -

   * - Street Canyon
     - :doc:`api/channel.cluster_delay_line_street_canyon`
     -

   * - Urban Macrocells
     - :doc:`api/channel.cluster_delay_line_urban_macrocells`
     -

   * - Multipath Fading
     - :doc:`api/channel.multipath_fading_channel`
     -

   * - Cost256
     - :class:`MultipathFadingCost256 <hermespy.channel.multipath_fading_templates.MultipathFadingCost256>`
     -

   * - 5G Tapped Delay Lines
     - :class:`MultipathFading5GTDL <hermespy.channel.multipath_fading_templates.MultipathFading5GTDL>`
     -

   * - Exponential
     - :class:`MultipathFadingExponential <hermespy.channel.multipath_fading_templates.MultipathFadingExponential>`
     -

   * - Quadriga
     - :doc:`api/channel.quadriga`
     -

   * - **Spatial Delay Channel**
     - :class:`SpatialDelayChannel<hermespy.channel.delay.SpatialDelayChannel>`
     - 

   * - **Random Delay Channel**
     - :class:`RandomDelayChannel<hermespy.channel.delay.RandomDelayChannel>`
     - 

   * - **Radar Single Reflector**
     - :class:`SingleTargetRadarChannel<hermespy.channel.radar_channel.SingleTargetRadarChannel>`
     - :doc:`examples/jcas`

   * - **Radar Multi Reflector**
     - :class:`MultiTargetRadarChannel<hermespy.channel.radar_channel.MultiTargetRadarChannel>`
     -

.. list-table:: **Communication Modulation**
   :header-rows: 1

   * - Waveform
     - API References
     - Examples

   * - Chirp FSK
     - :doc:`api/modem.waveform_chirp_fsk`
     - :doc:`examples/chirp_fsk_lora`
 
   * - OFDM
     - :doc:`api/modem.waveform_ofdm`
     - :doc:`examples/interference_ofdm_single_carrier`,
       :doc:`examples/ofdm_5g`,
       :doc:`examples/ofdm_single_carrier`

   * - Single Carrier
     - :doc:`api/modem.waveform_single_carrier`
     - 

   * - Root Raised Cosine
     - :doc:`api/modem.waveform_single_carrier`
     - :doc:`examples/chirp_qam`,
       :doc:`examples/hardware_model`,
       :doc:`examples/interference_ofdm_single_carrier`

   * - Raised Cosine
     - :doc:`api/modem.waveform_single_carrier`
     -  

   * - Rectangular
     - :doc:`api/modem.waveform_single_carrier`
     - 

   * - FMCW
     - :doc:`api/modem.waveform_single_carrier`
     - :doc:`examples/jcas`


.. list-table:: **Communication Receiver Algorithms**
   :header-rows: 1

   * - Algorithm
     - API References
     - Examples

   * - Synchronization
     - :class:`Synchronization <hermespy.modem.waveform_generator.Synchronization>`
     - 

   * - Channel Estimation
     - :class:`ChannelEstimation <hermespy.modem.waveform_generator.ChannelEstimation>`
       :class:`IdealChannelEstimation <hermespy.modem.waveform_generator.IdealChannelEstimation>`
       :class:`Single Carrier Least-Squares <hermespy.modem.waveform_single_carrier.SingleCarrierLeastSquaresChannelEstimation>`
       :class:`OFDM Least-Squares <hermespy.modem.waveform_generator_ofdm.OFDMLeastSquaresChannelEstimation>`
     -

   * - Equalization
     - :class:`ChannelEqualization <hermespy.modem.waveform_generator.ChannelEqualization>`
       :class:`Zero-Forcing <hermespy.modem.waveform_generator.ZeroForcingChannelEqualization>`
       :class:`OFDM MMSE <hermespy.modem.waveform_generator_ofdm.OFDMMinimumMeanSquareChannelEqualization>`
       :class:`Single Carrier MMSE <hermespy.modem.waveform_single_carrier.SingleCarrierMinimumMeanSquareChannelEqualization>`
     -



.. list-table:: **Sensing Modulation**
   :header-rows: 1

   * - Waveform
     - API References
     - Examples

   * - FMCW
     - :doc:`api/radar.fmcw`
     - 

   * - Matched Filter JCAS 
     - :doc:`api/jcas.matched_filtering`
     - :doc:`examples/jcas`


.. list-table:: **Multi Antenna Algorithms**
   :header-rows: 1

   * - Algorithm
     - API References
     - Examples

   * - Alamouti
     - :doc:`api/modem.precoding.space_time_block_coding`
     - 

   * - Ganesan
     - :doc:`api/modem.precoding.space_time_block_coding`
     - 

   * - 
     - :doc:`api/modem.precoding.single_carrier`
     - :doc:`examples/ofdm_5g`

   * - 
     - :doc:`api/modem.precoding.spatial_multiplexing`
     - 

   * - Maximum Ratio Combining
     - :doc:`api/modem.precoding.ratio_combining`
     - 


.. list-table:: **Precodings**
   :header-rows: 1

   * - Algorithm
     - API References
     - Examples

   * - DFT
     - :doc:`api/modem.precoding.dft`
     - 


.. list-table:: **Hardware Models**
   :header-rows: 1

   * - Model
     - API References
     - Examples

   * - Power Amplifier
     - :doc:`PA <api/simulation.rf_chain.power_amplifier>`,
       :class:`Clipping <hermespy.simulation.rf_chain.power_amplifier.ClippingPowerAmplifier>`,
       :class:`Rapp <hermespy.simulation.rf_chain.power_amplifier.RappPowerAmplifier>`,
       :class:`Saleh <hermespy.simulation.rf_chain.power_amplifier.SalehPowerAmplifier>`,
       :class:`Custom AM/AM AM/PM Response <hermespy.simulation.rf_chain.power_amplifier.RappPowerAmplifier>`
     - :doc:`examples/hardware_model`

   * - I/Q Imbalance
     - :doc:`/api/simulation.rf_chain`
     - :doc:`examples/hardware_model`

   * - Anlog Digital Conversion
     - :doc:`api/simulation.analog_digital_converter`
     - :doc:`examples/hardware_model`
    
   * - **Phase Noise**
     - :doc:`api/simulation.rf_chain.phase_noise`
     - :doc:`examples/hardware_model`

   * - Antenna Characteristics
     - :doc:`api/core.antennas`
     - 

   * - Antenna Arrays
     - :doc:`api/core.antennas`
     - 

   * - **Mutual Coupling**
     - :doc:`api/simulation.coupling`
       :doc:`api/simulation.coupling.impedance`
       :doc:`api/simulation.coupling.perfect`
     - 

   * - **Transmit-Receive Isolation**
     - :doc:`api/simulation.isolation`
       :doc:`api/simulation.isolation.perfect`
       :doc:`api/simulation.isolation.impedance`
       :doc:`api/simulation.isolation.specific`
     -

   * - Noise
     - :doc:`api/simulation.noise`
     -


.. list-table:: **Key Performance Indicators**
   :header-rows: 1

   * - Indicator
     - Evaluator
     - Examples

   * - Bit Error Rate
     - :class:`BitErrorEvaluator <hermespy.modem.evaluators.BitErrorEvaluator>`
     - 

   * - Block Error Rate
     - :class:`BlockErrorEvaluator <hermespy.modem.evaluators.BlockErrorEvaluator>`
     - 

   * - Frame Error Rate
     - :class:`FrameErrorEvaluator <hermespy.modem.evaluators.FrameErrorEvaluator>`
     - 

   * - Throughput
     - :class:`ThroughputEvaluator <hermespy.modem.evaluators.ThroughputEvaluator>`
     - 

   * - **Receiver Operating Charactersitic**
     - :class:`ReceiverOperatingCharacteristic <hermespy.radar.evaluators.ReceiverOperatingCharacteristic>`
     - 

   * - **Detection RMSE**
     - :class:`RootMeanSquareError <hermespy.radar.evaluators.RootMeanSquareError>`
     - 

.. list-table:: **Hardware Interfaces**
   :header-rows: 1

   * - Interface
     - API
     - Examples

   * - **Soundcard**
     - :doc:`/api/hardware_loop.audio.device.AudioDevice`
     - :doc:`/notebooks/audio`, :doc:`/examples/audio`

   * - **USRP**
     - :doc:`/api/hardware_loop.uhd.usrp.UsrpDevice`
     - :doc:`examples/uhd`

.. footbibliography::
