In this example we simulate FSK modulated chirps (similar to LORA).
A bandwidth of B = 500kHz is considered, with spreading factor SF = 8,
This corresponds to M = 2^SF = 256 different initial frequencies, spaced by
\delta_f = B / M = 1953.125Hz
The symbol rate (chirp duration) is given by Ts = 2^SF/BW = .512 ms
Data is uncoded, and the data rate is
SF * BW / 2 **SF = log2(M) / Ts = 15625 kbps

Frames have 160 bits, i.e., 20 FSK symbols.

A carrier frequency of 865MHz is considered, with Rayleigh fading and a speed
of 10m/s.
