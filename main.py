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


def single_test(freqs, bit_rates, send_bits=tests.testbits, **kwargs):
    print('Frequencies: {}'.format(freqs))
    print('Min freq: {}, Max freq: {}'.format(min(freqs), max(freqs)))
    print('bit_rates: {}'.format(bit_rates))

    hamming = True
    enc.encode('bin.wav', send_bits, freqs, bit_rates, hamming=hamming, plot_audio=False)
    am.playrec('bin.wav', '_bin.wav', plot_ideal_signal=False)
    ans = dec.decode('_bin.wav', bit_rates, len(send_bits), freqs, hamming=hamming, plot_sync=False, plot_main=False, plot_conv=True)

    error = utils.calc_error_per_freq(send_bits, ans, freqs, bit_rates)

    if kwargs.get('plot_error_graph'):
        utils.plot_smooth_error_graph(send_bits, ans)

    if list(ans) == list(send_bits):
        print("YEET!")
    else:
        print("SHITE")

    return ans


def transmit(freqs, bit_rates, send_bits, **kwargs):
    print('Frequencies: {}'.format(freqs))
    print('Min freq: {}, Max freq: {}'.format(min(freqs), max(freqs)))
    print('bit_rates: {}'.format(bit_rates))

    hamming = True
    enc.encode('bin.wav', send_bits, freqs, bit_rates, hamming=hamming, plot_audio=False)

    print('Duration: {}'.format(utils.get_wav_duration('bin.wav')))
    print('No of bits: {}'.format(len(send_bits)))

    input('>>Press enter to play')
    am.play_wav('bin.wav')


def recieve(freqs, bit_rates, **kwargs):
    print('Frequencies: {}'.format(freqs))
    print('Min freq: {}, Max freq: {}'.format(min(freqs), max(freqs)))
    print('bit_rates: {}'.format(bit_rates))

    hamming = True

    t = input('>>Enter recording duration: ')
    no_of_bits = input('>>Enter number of bits: ')
    input('>>Press enter to start')
    am.record('_bin.wav', t)
    ans = dec.decode('_bin.wav', bit_rates, no_of_bits, freqs, hamming=hamming, plot_sync=False, plot_main=False, plot_conv=False)

    return ans


# with open('img.jpg', 'rb') as f:
#     b = np.array(bytearray(f.read()))
#     bits = np.unpackbits(b)

freqs = [4000 + 2000*i for i in range(8)]
data_rates = [(400 + 200*i)for i in range(8)]
freqs.remove(16000)
data_rates.remove(1600)
print('{} bytes/s'.format(np.sum(data_rates)/8))
recieved_bits = single_test(freqs, data_rates, tests.testbits, plot_error_graph=True)

recieved_bytes = np.packbits(recieved_bits)
# with open('img2.jpg', 'wb') as f:
#     f.write(recieved_bytes)

# image = Image.open(io.BytesIO(recieved_bytes))
# image.save('img2.jpg')


plt.show()