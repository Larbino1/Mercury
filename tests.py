import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import time
import collections

import utils
import decoding as dec
import encoding as enc
from consts import *

# testbytes = np.fromstring('''80 01''', dtype=np.ubyte, count=2)

#testbytes = np.fromstring('''64 36 de 9c 7d c6 a0 76''', dtype=np.ubyte, count=8)

# testbytes = np.fromstring('''64 36 de 9c 7d c6 a0 76 07 8b 0d 78 2f 47 07 6c
# c0 bf 25 ed 9f 9a 9e e8 6e ff 5f b5 07 b6 57 05
# da 00 9a 62 ff 2f 0f e0 91 35 a4 cf 35 23 19 f2
# ''', dtype=np.ubyte, count=48)

# testbytes = np.fromstring('''64 36 de 9c 7d c6 a0 76 07 8b 0d 78 2f 47 07 6c
# c0 bf 25 ed 9f 9a 9e e8 6e ff 5f b5 07 b6 57 05
# da 00 9a 62 ff 2f 0f e0 91 35 a4 cf 35 23 19 f2
# 1c 23 61 a4 85 1c 22 7f 75 68 7f 85 98 e8 0b 06
# fb 0e e5 32 73 d8 d6 7b 81 d2 d1 dc 84 84 09 41
# 82 ae 26 09 45 40 13 ae 3e 84 17 c8 2d 7f 07 38
# 3a a5 cc f9 ''', dtype=np.ubyte, count=100)

l = 100
testbytes = np.fromstring(np.random.bytes(l), dtype=np.ubyte, count=l)

testbits = np.unpackbits(testbytes)


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


def record_f_er_br(am):
    min_freq = 1000
    max_freq = 24000
    freq_div = 180
    frequencies = [min_freq + (max_freq - min_freq) * i / freq_div for i in range(freq_div)]

    min_br = 10 * 8
    max_br = 100 * 8
    br_div = 20
    bit_rates = [min_br + (max_br - min_br) * i / br_div for i in range(br_div)]

    fr = []
    er = []
    br = []
    for freq in frequencies:
        freqs = [freq]
        for phy_bit_rate in bit_rates:
            print('Frequencies: {}'.format(freqs))
            print('Min freq: {}, Max freq: {}'.format(min(freqs), max(freqs)))
            print('bit_rate: {}'.format(phy_bit_rate))

            hamming = False
            enc.encode('bin.wav', testbits, freqs, phy_bit_rate, hamming=hamming)
            am.playrec('bin.wav', '_bin.wav', plot_ideal_signal=False)
            ans = dec.decode('_bin.wav', phy_bit_rate, len(testbits), freqs, hamming=hamming, plot_sync=False, plot_main=False)

            # print(ans)
            # print(list(tests.testbits))

            error = utils.calc_error(testbits, ans)
            # utils.plot_smooth_error_graph(tests.testbits, ans)

            if list(ans) == list(testbits):
                print("YEET!")
            else:
                print("SHITE")
            print('')

            fr.append(freqs[0])
            br.append(phy_bit_rate)
            er.append(error)

    with open('freq_vs_error.txt', 'a') as f:
        for i in zip(fr, er, br):
            f.write('{}, {}, {}\n'.format(i[0], i[1], i[2]))


def plot_f_er_br():
    with open('freq_vs_error.txt') as f:
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

        print(freqs)

        errors = [[0]*len(bit_rates)]*len(freqs)
        print(np.array(errors).shape)
        for freq, error, bit_rate in l2:
            # print('Frequency index: {}, bit rate index: {}'.format(freqs.get(freq), bit_rates.get(bit_rate)))
            errors[int(freqs.get(freq))][int(bit_rates.get(bit_rate))] = error

        CS = plt.contour(list(bit_rates), list(freqs), errors)
        plt.clabel(CS)
        plt.show()
