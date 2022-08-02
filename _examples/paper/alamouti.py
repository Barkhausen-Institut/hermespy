import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plt


data = 10 * np.exp(2j * pi * np.random.uniform(size=100))
channel = np.random.normal(size=(2, 2, 50))+ 1j *np.random.normal(size=(2, 2, 50))


alamouti_encoding = np.empty((2, len(data)), dtype=complex)
alamouti_encoding[0, :] = data
alamouti_encoding[1, 0::2] = -data[1::2].conj()
alamouti_encoding[1, 1::2] = data[0::2].conj()

propagation = np.empty((2, 100), dtype=complex)

for i in range(50):
    propagation[:, 2*i:2*(1+i)] = channel[:, :, i] @ alamouti_encoding[:, 2*i:2*(1+i)]

channel = channel[0, ::]
weight_norms = np.sum(np.abs(channel) ** 2, axis=0, keepdims=False)

decoded_symbols = np.empty(len(data), dtype=complex)
decoded_symbols[0::2] = (channel[0].conj() * propagation[0, 0::2] + channel[1] * propagation[0, 1::2].conj()) / weight_norms
decoded_symbols[1::2] = (channel[0].conj() * propagation[0, 1::2] - channel[1] * propagation[0, 0::2].conj()) / weight_norms

plt.scatter(data.real, data.imag)
plt.scatter(decoded_symbols.real, decoded_symbols.imag, marker='x')
plt.show()