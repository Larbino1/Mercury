import matplotlib.pyplot as plt
import numpy as np
import time
import random
import PIL.Image as Image
import io

import utils
import tests
import decoding as dec
import encoding as enc
from consts import *

am = utils.AudioManager()


# data, compression, freqs, coding, modulation
def single_test(data, compression, freqs, coding, modulation, **kwargs):

    enc.encode(data, compression, freqs, coding, modulation, **kwargs)
    am.playrec('bin.wav', '_bin.wav')
    ans = dec.decode(len(data), compression, freqs, coding, modulation, **kwargs)  # hamming plot_sync plot_main plot_conv

    error = utils.calc_error_per_freq(data, ans, freqs)

    if kwargs.get('plot_errors'):
        utils.plot_smooth_error_graph(data, ans)

    if list(ans) == list(data):
        print("YEET!")
    else:
        print("SHITE")

    return ans


def transmit(freqs, bit_rates, send_bits, **kwargs):
    # Compress ->bits

    # Freq Multiplex

    # Coding

    # Modulate

    print('Frequencies: {}'.format(freqs))
    print('Min freq: {}, Max freq: {}'.format(min(freqs), max(freqs)))
    print('bit_rates: {}'.format(bit_rates))

    hamming = False
    enc.encode('bin.wav', send_bits, freqs, bit_rates, hamming=hamming, plot_audio=False)

    print('Duration: {}'.format(utils.get_wav_duration('bin.wav')))
    print('No of bits: {}'.format(len(send_bits)))

    input('>>Press enter to play')
    am.play_wav('bin.wav')
    am.sd.wait()


def recieve(freqs, bit_rates, **kwargs):
    print('Frequencies: {}'.format(freqs))
    print('Min freq: {}, Max freq: {}'.format(min(freqs), max(freqs)))
    print('bit_rates: {}'.format(bit_rates))

    hamming = False

    t = float(input('>>Enter recording duration: '))
    no_of_bits = int(input('>>Enter number of bits: '))
    input('>>Press enter to start')
    am.record('_bin.wav', t)
    am.sd.wait()

    ans = dec.decode('_bin.wav', bit_rates, no_of_bits, freqs, hamming=hamming, plot_sync=False, plot_main=False, plot_conv=False)

    return ans


def send_jpg(freqs, bit_rates, filename, savename, **kwargs):
    with open(filename, 'rb') as f:
        b = np.array(bytearray(f.read()))
    bits = np.unpackbits(b)
    recieved_bits = single_test(freqs, bit_rates, bits, **kwargs)
    recieved_bytes = bytes(np.packbits(recieved_bits))
    with open(savename, 'wb') as f:
        f.write(recieved_bytes)


# scale_factor = 1.185
# freqs = [2000 * scale_factor**i for i in range(13)]
# data_rates = [200 * scale_factor**i for i in range(13)]
# print('{} bytes/s'.format(np.sum(data_rates)/8))

freqs = [3000]

recieved_bits = single_test(tests.testbits, 'None', freqs, 'none', 'psk', plot_sync=True, plot_audio=False, plot_conv=True, plot_complex=True, plot_errors=True, plot_filters=True)
# send_jpg(freqs, data_rates, 'test.png', 'test2.png', hamming=False, plot_audio=False, psk=True, plot_sync=False, plot_conv=False, plot_complex=True, plot_errors=True)
# transmit(freqs, data_rates, tests.testbits)
# recieve(freqs, data_rates)

plt.show()
