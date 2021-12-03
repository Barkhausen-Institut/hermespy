# Features

The current releases encompasses the following (new features in __bold__):

##  5G++ and Usability Release (Nov 2021)

* **Modulation and Coding**
  * generic PSK/QAM/PAM modem with square pulses, (root)-raised-cosine filters or FMCW
  * linear equalizers for non-orthogonal FMCW pulses in AWGN
  * LLR calculation for BPSK/QAM/16-/64-/256-QAM
  * chirp FSK 
  * generic OFDM with arbitrary allocation of data and reference symbols in each resource element
  * DFT-spread supported
  * __unified multicarrier framework using a GFDM precoder__
  * repetition and LDPC codes
  * __faster LDPC decoder__
  * __3GPP-like scrambler__
  * __block interleaver__
  * __CRC overhead__
  * transmit diversity (Alamouti) with 2 or 4 tx antennas
  * open-loop spatial multiplexing with linear receivers
  * receive diversity (SC or MRC)
  * __radar detection for FMCW__
  * __channel estimation for OFDM__

* **Channel and interference model**
  * time-variant multipath channel with arbitrary power delay profile, Rice/Rayleigh fading
  * COST-259 macrocell model
  * 5G TDL model
  * MIMO support with antenna correlation, following Kronecker model
  * Interface to Quadriga channel model (requires Matlab or Octave)
  * Interference among different modems, with arbitrary transmit powers for different transmitters
  * Support for transmitters using different carrier frequencies and bandwidths
  * __single-target__ radar channel model

* **RF Chain**
  * memoryless non linear power amplifier following an ideal clipper, Rapp's, Saleh's or any arbitrary AM/AM AM/PM responses
  * __random time offset__
  * __I/Q imbalance__

* **Simulation**
  * __Installation as a Python library__
  * __greater modularity and standalone usage of simulator classes__
  * Full configuration using __YAML__ settings files
  * Script for plotting different simulations in one graph
  * Drops containing several frames
  * Support for multiple transmitter and receivers
  * user manual provided

* **Statistics**
  * BER/FER statistics
  * confidence interval is calculated and may be considered as a stopping criterion
  * theoretical results available (depending on simulation scenario)
  * power spectral density and time-frequency analysis
  * __time-domain waveform plots__
  * __constellation plots__
  
## Release Plan
Full releases with a new set of features will be provided on a half-yearly basis, with software patches in between.
For the next release, the current plan is

### May 2022: "Full Radar-Release"

* **Modulation and Coding**
  * radar detection for FMCW and OFDM
  * beamforming

* **Channel and interference model**
  * HW in the loop

* **RF Chain**
  * phase noise
