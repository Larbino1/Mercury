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

# phy_bit_rate = 10 * 8

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

testbits = tests.testbits
freqs = [15000]
phy_bit_rate = 630
print('Frequencies: {}'.format(freqs))
print('Min freq: {}, Max freq: {}'.format(min(freqs), max(freqs)))
print('bit_rate: {}'.format(phy_bit_rate))

hamming = False
enc.encode('bin.wav', testbits, freqs, phy_bit_rate, hamming=hamming)
am.playrec('bin.wav', '_bin.wav', plot_ideal_signal=False)
ans = dec.decode('_bin.wav', phy_bit_rate, len(testbits), freqs, hamming=hamming, plot_sync=False, plot_main=True)

# print(ans)
# print(list(tests.testbits))

error = utils.calc_error(testbits, ans)
# utils.plot_smooth_error_graph(tests.testbits, ans)

if list(ans) == list(testbits):
    print("YEET!")
else:
    print("SHITE")
print('')

tests.plot_f_er_br()
