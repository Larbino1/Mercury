from utils import *
from consts import *

import wave
import math
import time
from itertools import zip_longest
import numpy as np
import matplotlib.pyplot as plt


def hamming_7_4(bit_array, auto_pad=True):
    # print("encoding hamming 7_4,\trecieved {} bits".format(len(bit_array)))
    if auto_pad:
        bit_array = pad(bit_array, 4)
    assert len(bit_array) % 4 == 0, 'For hamming 7:4 bit array must be multiple of 4'
    chunks = chunk(bit_array, 4)
    bit_array = np.empty_like([], dtype='bool')
    for l in chunks:
        p1 = (l[0]+l[1]+l[3]) % 2
        p2 = (l[0]+l[2]+l[3]) % 2
        p3 = (l[1]+l[2]+l[3]) % 2
        l = np.concatenate((l, [p1, p2, p3]))
        bit_array = np.concatenate((bit_array,l))
    # print("\t\t\t\t\t\treturned {} bits".format(len(bit_array)))
    return bit_array

#
# def hamming_8_4(bit_array):
#     hamming_7_4(bit_array)
#     p4 = sum(bit_array) % 2
#     return bit_array


def encode(data, compression, freqs, coding, modulation, **kwargs):
    print('Frequencies: {}'.format(freqs))
    print('Min freq: {}, Max freq: {}'.format(min(freqs), max(freqs)))

    # Compress ->bits
    if compression == 'image':
        print("Compression: Image")
        with open(kwargs.get('image_file'), 'rb') as f:
            b = np.array(bytearray(f.read()))
        send_bits = np.unpackbits(b)
    else:
        print("Compression: None")
        send_bits = data

    # Freq Multiplex
    bit_rates = get_data_rates(freqs)
    bit_streams = split_data_into_streams(send_bits, bit_rates)

    # Coding
    if coding == 'hamming':
        print("Coding: Hamming")
        coded_bit_streams = [hamming_7_4(stream) for stream in bit_streams]
    else:
        print("Coding: None")
        coded_bit_streams = bit_streams

    # Modulate
    # Add leading silence and sync pulse
    audio = []
    audio = append_silence(audio, duration_milliseconds=500)
    audio.extend(get_sync_pulse())

    audio_streams = []
    for stream, freq, rate in zip(coded_bit_streams, freqs, bit_rates):
        if modulation == "psk":
            print("Modulation: PSK")
            audio_streams.append(modulate_psk(stream, freq, rate))
        elif modulation == "simple":
            print("Modulation: Simple")
            audio_streams.append(modulate_simple(stream, freq, rate))
        else:
            raise Exception("Invalid modulation type: {}".format(modulation))
    audio.extend([np.mean(i) for i in zip(*audio_streams)])

    # Add trailing silence
    audio.extend(get_silence(duration_milliseconds=500))

    # Save file
    save_wav(audio, 'bin.wav')

    if kwargs.get('plot_audio'):
        plt.figure('audio')
        plt.plot(audio)
        plt.draw()


def modulate_psk(bit_array, frequency, data_rate):
    audio = []
    duration = 1000 / data_rate
    loss_of_sync_per_bit = SAMPLE_RATE / data_rate % 1
    # e.g value of 0.1 means we should append 40.1 samples but only append 40
    # Keep track of total lag, and compensate with added samples

    # Pad for PSK
    bit_array = pad(bit_array, 2)
    # Insert 16 bits detailing length of transmission
    assert len(bit_array) < 65536, "ERROR maximum packet size exceeded"
    bit_count_frame = "{:0>16b}".format(len(bit_array))
    print("Bit count frame: {}".format(bit_count_frame))
    assert len(bit_count_frame) == 16
    coded_bit_count_frame = hamming_7_4([int(i) for i in bit_count_frame], auto_pad=False)
    print("Coded bit count frame: {}".format(''.join([str(i) for i in coded_bit_count_frame])))
    bit_array = np.insert(bit_array, 0, coded_bit_count_frame)

    print("len(bit_array): {}".format(len(bit_array)))

    # Insert 8 bits detailing symbol space
    bit_array = np.insert(bit_array, 0, [1, 1, 1, 0, 0, 1, 0, 0])

    print(bit_array)

    cumulative_lag = 0
    chunks = chunk(bit_array, 2)
    for bit_pair in chunks:
        b1, b2 = bit_pair
        if b1 and b2:
            audio.extend(get_neg_coswave(frequency, duration))
        elif b1:
            audio.extend(get_neg_sinewave(frequency, duration))
        elif b2:
            audio.extend(get_coswave(frequency, duration))
        else:
            audio.extend(get_sinewave(frequency, duration))
        cumulative_lag += loss_of_sync_per_bit
        if cumulative_lag > 1:
            audio.append(audio[-1])
            cumulative_lag -= 1
    return audio


def modulate_simple(bit_array, frequency, data_rate):
    audio = []
    duration = 1000 / data_rate
    loss_of_sync_per_bit = SAMPLE_RATE / data_rate % 1
    # e.g value of 0.1 means we should append 40.1 samples but only append 40
    # Keep track of total lag, and compensate with added samples
    cumulative_lag = 0
    for bit in bit_array:
        if bit:
            audio.extend(get_sinewave(frequency, duration))
        else:
            audio.extend(get_silence(duration))
        cumulative_lag += loss_of_sync_per_bit
        if cumulative_lag > 1:
            audio.append(audio[-1])
            cumulative_lag -= 1
    return audio

