import matplotlib.pyplot as plt
import numpy as np
import time
import random

import utils
import tests
import decoding as dec
import encoding as enc
from consts import *

am = utils.AudioManager()


def play_and_record(file_to_play, file_to_save):
    duration = utils.get_wav_duration(file_to_play)

    am.playrec(file_to_play, file_to_save, duration, plot_ideal_signal=False)

phy_bit_rate = 60 * 8
base_freq = 4000
freqs = [base_freq + 250*i for i in range(8)]

print('Frequencies: {}'.format(freqs))
print('Min freq: {}, Max freq: {}'.format(min(freqs), max(freqs)))

hamming = True
enc.encode('bin.wav', tests.testbits, freqs, phy_bit_rate, hamming=hamming)
play_and_record('bin.wav', '_bin.wav')
ans = dec.decode('_bin.wav', phy_bit_rate, len(tests.testbits), freqs, hamming=hamming, plot_sync=False, plot_main=False)

print(ans)
print(list(tests.testbits))

utils.calc_error(tests.testbits, ans)

plt.figure('errors')
errors = np.bitwise_xor(ans, list(tests.testbits))
sigma = 100
x = np.linspace(-3*sigma, 3*sigma, 6*sigma)
gaussian = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * sigma**2))
smooth = np.convolve(gaussian, errors)
plt.plot(smooth)
plt.show()

if list(ans) == list(tests.testbits):
    print("YEET!")
else:
    print("SHITE")

# ans = np.copy(tests.testbits)
# print(ans)
# ans = enc.hamming_7_4(ans)
# print(ans)
# for n in range(len(ans)):
#     if random.random() > 0.97:
#         ans[n] = inv[ans[n]]
# # ans[2] = inv[ans[2]]
# print(ans)
# ans = dec.hamming_7_4(ans)
# print(ans)
#
# utils.calc_error(tests.testbits, ans)


def hamming_test():
    for i in range(16):
        test = np.array([int(x) for x in format(i, '004b')])
        ans = enc.hamming_7_4(test)
        for n in range(len(ans)):
            x = np.copy(ans)
            print(test)
            print(x)
            x[n] = inv[x[n]]
            print(x)
            x = dec.hamming_7_4(x)
            print(x)
            for i in zip(x,test):
                assert i[0] == i[1]
            print('')
            print('')
        print(test)
