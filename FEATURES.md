# Features

The current releases encompasses the following:
## 5G-Ready Release (May 2021)

* **Modulation and Coding**
  * generic PSK/QAM/PAM modem with square pulses, (root)-raised-cosine filters or FMCW
  * linear equalizers for non-orthogonal FMCW pulses in AWGN
  * LLR calculation for BPS/QAM/16-/64-/256-QAM
  * chirp FSK 
  * generic OFDM with arbitrary allocation of data and reference symbols in each resource element (reference currently a random signal)
  * DFT spread supported
  * repetition and LDPC codes
  * transmit diversity (Alamouti) with 2 or 4 tx antennas
  * open-loop spatial multiplexing with linear receivers
  * receive diversity (SC or MRC)

* **Channel and interference model**
  * time-variant multipath channel with arbitrary power delay profile, Rice/Rayleigh fading
  * COST-259 macrocell model
  * 5G TDL model
  * MIMO support with antenna correlation, following Kronecker model
  * Interface to Quadriga channel model (requires Matlab or Octave)
  * Interference among different modems, with arbitrary transmit powers for different transmitters
  * Support for transmitters using different carrier frequencies and bandwidths

* **RF Chain**
  * memoryless non linear power amplifier following an ideal clipper, Rapp's, Saleh's or any arbitrary AM/AM AM/PM responses

* **Simulation**
  * Full configuration using settings files
  * Script for plotting different simulations in one graph
  * Drops containing several frames
  * Support for multiple transmitter and receivers
  * user manual provided

* **Statistics**
  * BER/FER statistics
  * confidence interval is calculated and may be considered as a stopping criterion
  * theoretical results available (depending on simulation scenario)
  * power spectral density and time-frequency analysis
  
## Release Plan
Full releases with a new set of features will be provided on a half-yearly basis, with software patches in between.
For the next release, the current plan is

### October 2021 / "5G++/Radar-Release (October 2021)"

* **Modulation and Coding**
  * Fkexible GFDM-Modem
  * improved-performance (faster) LDPC decoding
  * equalization for SC modulation
  * channel estimation for OFDM
  * radar detection for FMCW and OFDM

* **RF Chain**
  * IQ imbalance
  * phase noise
