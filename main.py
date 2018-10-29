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

    hamming = True
    enc.encode('bin.wav', testbits, freqs, bit_rates, hamming=hamming, plot_audio=False)
    am.playrec('bin.wav', '_bin.wav', plot_ideal_signal=False)
    ans = dec.decode('_bin.wav', bit_rates, len(testbits), freqs, hamming=hamming, plot_sync=False, plot_main=False, plot_conv=False)

    error = utils.calc_error_per_freq(testbits, ans, freqs, bit_rates)

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
#
# split = enc.split_data_into_streams(tests.testbits, [1])
# decoded = np.concatenate(split)
# print(*split)
# print(tests.testbits)
# print(decoded)
# utils.assert_arrays_equal(decoded, tests.testbits)

# single_test([6000, 14000], [300, 700], plot_error_graph=True)
# single_test([6000, 7000, 8000, 14000], [600,700,800, 800], plot_error_graph=True)

freqs = [4000 + 2000*i for i in range(8)]
data_rates = [(400 + 200*i)for i in range(8)]
freqs.remove(16000)
data_rates.remove(1600)
print('{} bytes/s'.format(np.sum(data_rates)/8))
single_test(freqs, data_rates, plot_error_graph=True)

plt.show()

utils.get_bandpass(400, SAMPLE_RATE)
