from utils import *
from consts import *
import log

import wave
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import tests

import encoding as enc


def hamming_7_4(bit_array, auto_unpad=True):
    log.debug("decoding hamming 7_4,\trecieved {} bits".format(len(bit_array)))
    from consts import inv
    assert len(bit_array) % 7 == 0, 'For hamming 7:4 decoding array must be multiple of 7'
    chunks = chunk(bit_array, 7)
    bit_array = np.empty_like([], dtype='bool')
    for l in chunks:
        p1_err, p2_err, p3_err = False, False, False
        if l[4] != ((l[0]+l[1]+l[3]) % 2):
            p1_err = True
        if l[5] != ((l[0]+l[2]+l[3]) % 2):
            p2_err = True
        if l[6] != ((l[1]+l[2]+l[3]) % 2):
            p3_err = True

        if p1_err and p2_err and p3_err:
            l[3] = inv[l[3]]
        elif p1_err and p2_err:
            l[0] = inv[l[0]]
        elif p1_err and p3_err:
            l[1] = inv[l[1]]
        elif p2_err and p3_err:
            l[2] = inv[l[2]]

        x = l[:4]
        bit_array = np.concatenate((bit_array, x))
    if auto_unpad:
        bit_array = unpad(bit_array, 4)
    log.debug("\t\t\t\t\t\treturned {} bits".format(len(bit_array)))
    return bit_array


# def hamming_8_4(bit_array):
#     # print("decoding hamming 7_4,\trecieved {} bits".format(len(bit_array)))
#     from consts import inv
#     assert len(bit_array) % 8 == 0, 'For hamming 7:4 decoding array must be multiple of 7'
#     chunks = chunk(bit_array, 8)
#     bit_array = np.empty_like([], dtype='bool')
#     for l in chunks:
#         p1_err, p2_err, p3_err, p4_err = 0, 0, 0, 0
#         if l[4] != ((l[0]+l[1]+l[3]) % 2):
#             p1_err = 1
#         if l[5] != ((l[0]+l[2]+l[3]) % 2):
#             p2_err = 1
#         if l[6] != ((l[1]+l[2]+l[3]) % 2):
#             p3_err = 1
#         if l[7] != (np.sum(l[:7]) % 2):
#             p4_err = 1
#
#         if p1_err and p2_err and p3_err:
#             l[3] = inv[l[3]]
#         elif p1_err and p2_err:
#             l[0] = inv[l[0]]
#         elif p1_err and p3_err:
#             l[1] = inv[l[1]]
#         elif p2_err and p3_err:
#             l[2] = inv[l[2]]
#
#         if sum([p1_err, p2_err, p3_err, p4_err]) % :
#             raise Exception("Double bit flip detected in hamming 8 4")
#
#         x = l[:4]
#         bit_array = np.concatenate((bit_array, x))
#     bit_array = unpad(bit_array, 4)
#     # print("\t\t\t\t\t\treturned {} bits".format(len(bit_array)))
#     return bit_array


def decode(bit_count, compression, freqs, coding, modulation, **kwargs):
    bit_rates = get_data_rates(freqs)
    assert len(freqs) == len(bit_rates), 'Must have same number of specified frequencies and data rates'

    # TODO Fix
    if coding == 'hamming':
        bit_count = bit_count * 7//4
        # Test data for plotting
        test_bit_streams = split_data_into_streams(enc.hamming_7_4(tests.testbits, auto_pad=True), bit_rates)
    else:
        test_bit_streams = split_data_into_streams(tests.testbits, bit_rates)

    with wave.open('rec/' + 'bin.wav') as f:
        audio = f.readframes(-1)
    audio = np.frombuffer(audio, dtype='int16')

    # Demodulate
    coded_bit_streams = demodulate(bit_count, audio, freqs, bit_rates, modulation, **kwargs)

    bit_streams = []
    for stream in coded_bit_streams:
        # Decode
        if coding == 'hamming':
            log.info("Decoding: Hamming")
            bit_streams.append(hamming_7_4(stream))
        else:
            log.info("Decoding: None")
            bit_streams = coded_bit_streams

    # Join streams
    ret = []
    for stream in bit_streams:
        ret.extend(stream)

    # Decompress
    log.info("Decompression: None")

    return ret


def demodulate(bit_count, signal, freqs, bit_rates, modulation, **kwargs):
    # Find start
    i_best = find_start_sample(signal, **kwargs)
    signal = signal[i_best:]

    # Plot signal after sync
    if kwargs.get('plot_main'):
        plt.figure('main')
        plt.plot(signal)
        plt.draw()

    # Calculate bit lengths of each stream
    stream_lengths = get_split_stream_lengths(bit_count, bit_rates)

    bit_streams = []
    for freq, bit_rate, stream_length in zip(freqs, bit_rates, stream_lengths):
        if modulation == 'psk':
            log.info("Demodulation: PSK")
            bit_streams.append(demodulate_psk(signal, freq, bit_rate, bit_count, **kwargs))
        elif modulation == 'simple':
            log.info("Demodulation: Simple")
            bit_streams.append(demodulate_simple(signal, freq, bit_rate, bit_count, **kwargs))

    return bit_streams


def generate_bit_centres(bit_rate, bit_count=-1):
    """ Generates the sample number for the centre of each symbol for PSK"""
    n = 0
    while True:
        ctr = round((n + 0.5) * SAMPLE_RATE / bit_rate)
        yield ctr
        n += 1
        if 0 < bit_count < n*2:
            break


def get_psk_magnitudes(conv, no_of_bits, phase_shift, bit_centre_generator, plot=False):
    ret = []
    for i in range(no_of_bits):
        bit_ctr = next(bit_centre_generator)
        sin_mag = conv[bit_ctr]
        cos_mag = conv[bit_ctr + phase_shift]
        ret.append((sin_mag, cos_mag))
        if plot:
            plot_complex(sin_mag, cos_mag, 'fig1', 'rx')
    return ret


def get_transform_matrix(magnitudes):
    """"
    Gets a least squares solution to mapping the first four symbols
    to known true values.
    """
    a = []
    for sin_mag, cos_mag in magnitudes[:4]:
        a.append([sin_mag, cos_mag])
    a = np.asarray(a)
    b = np.asarray([[0, -1], [-1, 0], [0, 1], [1, 0]])
    x, _c, _d, _e = np.linalg.lstsq(a, b)
    return x


def get_psk_symbol_stream(magnitudes, plot=False):
    """
    Takes the transformed magnitudes (i.e mapped onto 1,0 etc) and returns
    a list of symbols: 'a', 'b', 'c', 'd'
    """
    symbol_stream = []
    for i, mags in enumerate(magnitudes):
        sin_mag, cos_mag = mags
        if sin_mag > cos_mag:
            if cos_mag > -sin_mag:
                symbol_stream.append('a')
                graph_format = 'rx'
            else:
                symbol_stream.append('b')
                graph_format = 'gx'
        else:
            if cos_mag > -sin_mag:
                symbol_stream.append('c')
                graph_format = 'bx'
            else:
                symbol_stream.append('d')
                graph_format = 'yx'

        if plot:
            plt.figure('complex2')
            plt.plot(sin_mag, cos_mag, graph_format)
    return symbol_stream


def demodulate_psk(signal, freq, bit_rate, bit_count, **kwargs):
    bit_pair_centres = generate_bit_centres(bit_rate)
    # No of samples in quarter wavelength shift
    phase_shift = round(SAMPLE_RATE / freq / 4)

    filter = get_bandpass(freq, SAMPLE_RATE, **kwargs)
    conv = np.convolve(filter, signal, mode='same')

    if kwargs.get('plot_conv'):
        plot_psk_conv(freq, conv, signal, generate_bit_centres(bit_rate, bit_count + 8 + 16*7//4), phase_shift)

    # Decode data frame
    df_magnitudes = get_psk_magnitudes(conv, 4 + 8*7//4, phase_shift, bit_pair_centres, plot=kwargs.get('plot_complex'))
    x = get_transform_matrix(df_magnitudes)
    tfd_df_magnitudes = np.matmul(df_magnitudes, x)
    df_symbol_stream = get_psk_symbol_stream(tfd_df_magnitudes, kwargs.get('plot_complex'))

    # Build map between symbols and bits
    for y in range(4):
        for z in range(4):
            if y != z:
                if df_symbol_stream[y] == df_symbol_stream[z]:
                    plt.show()
                    raise Exception('Same symbol mapped twice')

        symbol_map = dict()
        symbol_map[df_symbol_stream[0]] = [1, 1]
        symbol_map[df_symbol_stream[1]] = [1, 0]
        symbol_map[df_symbol_stream[2]] = [0, 1]
        symbol_map[df_symbol_stream[3]] = [0, 0]

    coded_df_bits = []
    for symbol in df_symbol_stream[4:]:
        coded_df_bits.extend(symbol_map[symbol])
    log.debug("Coded bit count frame: {}".format(''.join([str(i) for i in coded_df_bits])))
    df_bits = hamming_7_4(coded_df_bits, auto_unpad=False)
    bit_count = int(''.join([str(bit) for bit in df_bits]), 2)

    log.debug("Data frame bits: {}".format(''.join([str(i) for i in df_bits])))
    log.debug("no of bits to recieve: {}".format(bit_count))

    assert bit_count % 2 == 0, 'Error in recieved bit count, must be even for psk after padding'
    symbol_count = bit_count//2

    data_magnitudes = get_psk_magnitudes(conv, symbol_count, phase_shift, bit_pair_centres)
    tfd_data_magnitudes = np.matmul(data_magnitudes, x)
    data_symbol_stream = get_psk_symbol_stream(tfd_data_magnitudes, kwargs.get('plot_complex'))

    data_stream = []
    for symbol in data_symbol_stream:
        data_stream.extend(symbol_map[symbol])

    # Unpad for PSK
    bit_stream = unpad(data_stream, 2)

    return bit_stream


def plot_psk_conv(freq, conv, signal, bit_pair_centres, phase_shift):
    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    ax1.set_title('Convolution for freq = {}'.format(str(freq)))
    ax2.set_title('Signal'.format(str(freq)))
    ax1.plot(conv)
    ax2.plot(signal)
    for i, bit_ctr in enumerate(bit_pair_centres):
        ax1.axvline(x=bit_ctr, color='g')
        ax2.axvline(x=bit_ctr, color='g')
        ax1.axvline(x=bit_ctr + phase_shift, color='r')
        ax2.axvline(x=bit_ctr + phase_shift, color='r')


def plot_complex(sin_mag, cos_mag, fig, graph_format='rx'):
    if fig == 'fig1':
        plt.figure('complex1')
        plt.title('First 4 symbols')
    elif fig == 'fig2':
        plt.figure('complex2')
        plt.title('All complex points')
    else:
        raise Exception("Invalid fig")
    plt.xlabel('R')
    plt.ylabel('I')
    plt.grid(True)
    plt.axes().set_aspect('equal', 'datalim')
    plt.plot(sin_mag, cos_mag, graph_format)


def find_start_sample(signal, **kwargs):
    sync_pulse = get_sync_pulse()
    sig = signal[:SAMPLE_RATE * 2]
    conv = np.convolve(sync_pulse, sig, mode='full')
    i_best = np.argmax(conv)
    if kwargs.get('plot_sync'):
        # Plot sync
        plt.figure('sync')
        # plt.plot(conv / max(conv), color='y')
        plt.plot(sig, color='r')
        plt.axvline(x=i_best, color='g')
    return i_best


def demodulate_simple(signal, freq, bit_rate, bit_count, **kwargs):
    # Convolve and store conv values at bit boundaries
    # bit_width = SAMPLE_RATE / rate
    bit_ctrs = [round((n+0.4) * SAMPLE_RATE/bit_rate) for n in range(bit_count)]
    duration = 1000 / bit_rate
    filter = get_bandpass(freq, SAMPLE_RATE, **kwargs)
    conv = np.convolve(filter, signal, mode='same')
    # conv = conv[len(filter)//2:]
    if kwargs.get('plot_conv'):
        plot_simple_conv(freq, bit_ctrs, conv)

    conv_values = [np.sum(abs(conv[bit_ctr-3:bit_ctr+3])) for i, bit_ctr in enumerate(bit_ctrs)]

    if kwargs.get('plot_conv') or kwargs.get('plot_main'):
        plt.draw()

    # Thresholding
    bit_stream = []
    threshold = np.mean(conv_values)
    for conv_value in conv_values:
        if conv_value > threshold:
            bit_stream.append(1)
        else:
            bit_stream.append(0)

    return bit_stream


def plot_simple_conv(freq, bit_ctrs, conv):
    plt.figure('conv for freq = {}'.format(str(freq)))
    plt.plot(conv)
    for i, bit_ctr in enumerate(bit_ctrs):
        plt.figure('conv for freq = {}'.format(str(freq)))
        plt.axvline(x=bit_ctr, color='r')