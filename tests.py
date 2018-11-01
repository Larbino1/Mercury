import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import time
import collections

import utils
import decoding as dec
import encoding as enc
from consts import *

l = 100

testbytes = np.fromstring(np.random.bytes(l), dtype=np.ubyte, count=l)
testbits = np.unpackbits(testbytes)
# testbytes = np.fromstring('Hello World', dtype=np.ubyte)
# testbits = np.unpackbits(testbytes)


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
            for i in zip(x, test):
                assert i[0] == i[1]
            print('')
            print('')
        print(test)


def record_f_er_br(am, filename, freq_params, br_params):
    min_freq, max_freq, freq_div = freq_params
    min_br, max_br, br_div = br_params

    frequencies = [min_freq + (max_freq - min_freq) * i / freq_div for i in range(freq_div)]
    bit_rates = [min_br + (max_br - min_br) * i / br_div for i in range(br_div)]

    fr = []
    er = []
    br = []
    for freq in frequencies:
        for phy_bit_rate in bit_rates:
            print('Frequency: {}'.format(freq))
            print('bit_rate: {}'.format(phy_bit_rate))

            hamming = False
            enc.encode('bin.wav', testbits, [freq], phy_bit_rate, hamming=hamming)
            am.playrec('bin.wav', '_bin.wav', plot_ideal_signal=False)
            ans = dec.decode('_bin.wav', phy_bit_rate, len(testbits), [freq], hamming=hamming, plot_sync=False, plot_main=False)

            # print(ans)
            # print(list(tests.testbits))

            error = utils.calc_error(testbits, ans)
            # utils.plot_smooth_error_graph(tests.testbits, ans)

            if list(ans) == list(testbits):
                print("YEET!")
            else:
                print("SHITE")
            print('')

            fr.append(freq)
            br.append(phy_bit_rate)
            er.append(error)

    with open(filename, 'a') as f:
        for i in zip(fr, er, br):
            f.write('{}, {}, {}\n'.format(i[0], i[1], i[2]))


def load_f_br_er(filename):
    with open(filename) as f:
        l = f.readlines()
        l2 = []
        for item in l:
            item = item.strip().split(', ')
            l2.append(item)
        fr, er, br = zip(*l2)
        freqs = dict()
        bit_rates = dict()
        n = 0
        for item in fr:
            if not freqs.get(item):
                freqs[item] = str(n)
                n += 1
        n = 0
        for item in br:
            if not bit_rates.get(item):
                bit_rates[item] = str(n)
                n += 1

        print(freqs.keys())

        errors = [[0]*len(bit_rates)]*len(freqs)
        print(np.array(errors).shape)
        for freq, error, bit_rate in l2:
            # print('Frequency index: {}, bit rate index: {}'.format(freqs.get(freq), bit_rates.get(bit_rate)))
            errors[int(freqs.get(freq))][int(bit_rates.get(bit_rate))] = error
        return freqs, bit_rates, errors


def plot_f_br_er(filename):
    freqs, bit_rates, errors = load_f_br_er(filename)
    CS = plt.contour(list(bit_rates), list(freqs), errors)
    plt.clabel(CS)
    plt.show()


def plot_er_br(filename, freq):
    freqs, bit_rates, errors = load_f_br_er(filename)
    freq_index = int(freqs[str(freq)])

    # # NORMALISE BIT RATES AGAINS FREQ
    # bit_rates = [float(freq)/float(br) for br in bit_rates]

    plt.plot(list(bit_rates), list(errors[freq_index]))
    plt.show()
