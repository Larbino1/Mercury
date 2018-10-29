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
# utils.calc_error(tests.testbits, ans)


def single_test(freqs, bit_rates, testbits=tests.testbits, **kwargs):
    print('Frequencies: {}'.format(freqs))
    print('Min freq: {}, Max freq: {}'.format(min(freqs), max(freqs)))
    print('bit_rate: {}'.format(bit_rates))

    hamming = False
    enc.encode('bin.wav', testbits, freqs, bit_rates, hamming=hamming, plot_audio=True)
    am.playrec('bin.wav', '_bin.wav', plot_ideal_signal=False)
    ans = dec.decode('_bin.wav', bit_rates, len(testbits), freqs, hamming=hamming, plot_sync=True, plot_main=True, plot_conv=True)

    error = utils.calc_error(testbits, ans)
    if kwargs.get('plot_error_graph'):
        utils.plot_smooth_error_graph(tests.testbits, ans)

    if list(ans) == list(testbits):
        print("YEET!")
    else:
        print("SHITE")


# filename = 'freq_vs_error4.txt'
# freq_params = (18000, 1000, 1)
# br_params = (100, 2000, 760*2)
# #tests.record_f_er_br(am, filename, freq_params, br_params)
# tests.plot_er_br(filename, '18000.0')

# for i in range(10):
#     freq = random.choice(freqs)
#     tests.plot_er_br(filename, freq)

# split = enc.split_data_into_streams(tests.testbits, [1])
# decoded = np.concatenate(split)
# print(*split)
# print(tests.testbits)
# print(decoded)
# utils.assert_arrays_equal(decoded, tests.testbits)

single_test([18000], [900], plot_error_graph=True)
plt.show()