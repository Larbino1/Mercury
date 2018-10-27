from utils import append_sinewaves, append_silence, get_sync_pulse, save_wav
from utils import chunk
from consts import *

import wave
import math
import time
import numpy as np
import matplotlib.pyplot as plt


def hamming_7_4(bit_array):
    print("encoding hamming 7_4,\trecieved {} bits".format(len(bit_array)))
    assert len(bit_array) % 4 == 0, 'For hamming 7:4 bit array must be multiple of 4'
    chunks = chunk(bit_array, 4)
    bit_array = np.empty_like([], dtype='bool')
    for l in chunks:
        p1 = (l[0]+l[1]+l[3]) % 2
        p2 = (l[0]+l[2]+l[3]) % 2
        p3 = (l[1]+l[2]+l[3]) % 2
        l = np.concatenate((l, [p1, p2, p3]))
        bit_array = np.concatenate((bit_array,l))
    print("\t\t\t\t\t\treturned {} bits".format(len(bit_array)))
    return bit_array


def encode(filename, bits_array, freqs, rate=25, **kwargs):
    # print('Encoding: {}'.format(bits_array))
    print('Encoding')
    total_bits = len(bits_array)
    # print('total_bits = {}'.format(total_bits))

    if kwargs.get('hamming'):
        bits_array = hamming_7_4(bits_array)

    audio = []
    audio = append_silence(audio, duration_milliseconds=500)
    audio.extend(get_sync_pulse())

    n = len(freqs)
    duration = 1000 / rate * n
    print('duration: {}'.format(duration))

    chunks = chunk(bits_array, n)
    for item in chunks:
        active_freqs = []
        for x, bit in enumerate(item):
            if bit:
                active_freqs.append(freqs[x])
        # print('{}/{}'.format(i, total_bits))
        audio = append_sinewaves(audio, freqs=active_freqs, duration_milliseconds=duration)

    audio = append_silence(audio, duration_milliseconds=500)

    save_wav(audio, filename)