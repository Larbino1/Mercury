from utils import *
from consts import *

import wave
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import tests

import encoding as enc


def hamming_7_4(bit_array):
    print("decoding hamming 7_4,\trecieved {} bits".format(len(bit_array)))
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
    print("\t\t\t\t\t\treturned {} bits".format(len(bit_array)))
    return bit_array


def decode(filename, bit_rates, bit_count, freqs, **kwargs):
    print('Decoding')

    #Check args
    bit_rates = check_given_rates(bit_rates, len(freqs))
    assert len(freqs) == len(bit_rates), 'Must have same number of specified frequencies and data rates'

    if kwargs.get('hamming'):
        bit_count = bit_count * 7//4
        # Test data for plotting
        test_bit_streams = split_data_into_streams(enc.hamming_7_4(tests.testbits), bit_rates)
    else:
        test_bit_streams = split_data_into_streams(tests.testbits, bit_rates)

    with wave.open('rec/' + filename) as f:
        signal = f.readframes(-1)
        signal = np.frombuffer(signal, dtype='int16')
        #print('bit_width {}'.format(bit_width))

        # Find start
        sync_pulse = get_sync_pulse()
        # sync_pulse *= np.blackman(len(sync_pulse))
        sig = signal[:SAMPLE_RATE*2]
        conv = np.convolve(sync_pulse, sig, mode='full')
        i_best = np.argmax(conv)
        if kwargs.get('plot_sync'):
            # Plot sync
            plt.figure('sync')
            # plt.plot(conv / max(conv), color='y')
            plt.plot(sig, color='r')
            plt.axvline(x=i_best, color='g')
        signal = signal[i_best:]

        # Plot signal after sync
        if kwargs.get('plot_main'):
            plt.figure('main')
            plt.plot(signal)
            plt.draw()

        if kwargs.get('psk'):
            ret = decode_psk(signal, bit_rates, bit_count, freqs, test_bit_streams, **kwargs)
        else:
            ret = decode_simple(signal, bit_rates, bit_count, freqs, test_bit_streams, **kwargs)

        if kwargs.get('hamming'):
            ret = hamming_7_4(ret)

        return ret


def decode_psk(signal, bit_rates, bit_count, freqs, test_bit_streams, **kwargs):
    # Convolve and store conv values at bit boundaries
    stream_lengths = get_split_stream_lengths(bit_count, bit_rates)
    symbol_streams = {freq: [] for freq in freqs}
    bit_streams = dict()
    for stream_length, freq, rate, test_bit_stream in zip(stream_lengths, freqs, bit_rates, test_bit_streams):
        bit_pair_centres = [round((n + 0.4) * SAMPLE_RATE/rate) for n in range((stream_length+8)//2)]
        phase_shift = round(SAMPLE_RATE/freq/4)
        # print('phaseshift = {}'.format(phase_shift))
        filter = get_bandpass(freq, SAMPLE_RATE, **kwargs)
        conv = np.convolve(filter, signal, mode='same')
        if kwargs.get('plot_conv'):
            pass
            plt.figure('conv for freq = {}'.format(str(freq)))
            plt.plot(conv)
            for i, bit_ctr in enumerate(bit_pair_centres):
                plt.figure('conv for freq = {}'.format(str(freq)))
                plt.axvline(x=bit_ctr, color='g')
                plt.axvline(x=bit_ctr + phase_shift, color='r')

        magnitudes = []
        for i, bit_ctr in enumerate(bit_pair_centres):
            sin_mag = conv[bit_ctr]
            cos_mag = conv[bit_ctr+phase_shift]
            magnitudes.append((sin_mag, cos_mag))
            if kwargs.get('plot_complex'):
                if i < 4:
                    plt.figure('complex1')
                    plt.xlabel('R')
                    plt.ylabel('I')
                    plt.title('Complex points')
                    plt.grid(True)
                    plt.axes().set_aspect('equal', 'datalim')
                    plt.plot(sin_mag, cos_mag, 'rx')

        a = []
        for sin_mag, cos_mag in magnitudes[:4]:
            a.append([sin_mag, cos_mag])
        a = np.asarray(a)
        b = np.asarray([[0, -1], [-1, 0], [0, 1], [1, 0]])
        x, _c, _d, _e = np.linalg.lstsq(a, b)
        magnitudes = np.matmul(magnitudes, x)
        # print(a)
        # print(np.matmul(a, x))

        if kwargs.get('plot_complex'):
            plt.figure('complex')
            plt.axes().set_aspect('equal', 'datalim')
            plt.figure('complex1')
            plt.axes().set_aspect('equal', 'datalim')
            for i, mags in enumerate(magnitudes):
                sin_mag, cos_mag = mags
                plt.figure('complex')
                if i < 4:
                    plt.figure('complex1')
                if sin_mag > cos_mag:
                    if cos_mag > -sin_mag:
                        symbol_streams[freq].append('a')
                        plt.plot(sin_mag, cos_mag, 'rx')
                    else:
                        symbol_streams[freq].append('b')
                        plt.plot(sin_mag, cos_mag, 'gx')
                else:
                    if cos_mag > -sin_mag:
                        symbol_streams[freq].append('c')
                        plt.plot(sin_mag, cos_mag, 'bx')
                    else:
                        symbol_streams[freq].append('d')
                        plt.plot(sin_mag, cos_mag, 'yx')
        else:
            for sin_mag, cos_mag in magnitudes:
                if sin_mag > cos_mag:
                    if cos_mag > -sin_mag:
                        symbol_streams[freq].append('a')
                    else:
                        symbol_streams[freq].append('b')
                else:
                    if cos_mag > -sin_mag:
                        symbol_streams[freq].append('c')
                    else:
                        symbol_streams[freq].append('d')

        for y in range(4):
            for z in range(4):
                if y != z:
                    if symbol_streams[freq][y] == symbol_streams[freq][z]:
                        plt.show()
                        raise Exception('Same symbol mapped twice')

        symbol_map = dict()
        symbol_map[symbol_streams[freq][0]] = [1, 1]
        symbol_map[symbol_streams[freq][1]] = [1, 0]
        symbol_map[symbol_streams[freq][2]] = [0, 1]
        symbol_map[symbol_streams[freq][3]] = [0, 0]

        bit_streams[freq] = []
        for symbol in symbol_streams[freq][4:]:
            bit_streams[freq].extend(symbol_map[symbol])

    if kwargs.get('plot_conv') or kwargs.get('plot_main'):
        plt.draw()

    # # Remove padding bits
    # for freq, stream in bit_streams:
    #     no_of_padding_bits = int(stream[:1], 2)
    #     stream = stream[1:-(1+no_of_padding_bits)]

    ret = []
    for stream in bit_streams.values():
        ret.extend(stream)
    print(ret)
    return ret


def decode_simple(signal, bit_rates, bit_count, freqs, test_bit_streams, **kwargs):
    # Convolve and store conv values at bit boundaries
    stream_lengths = get_split_stream_lengths(bit_count, bit_rates)
    conv_values = {freq: [] for freq in freqs}
    plot_main_flag = True
    for stream_length, freq, rate, test_bit_stream in zip(stream_lengths, freqs, bit_rates, test_bit_streams):
        # bit_width = SAMPLE_RATE / rate
        # TODO proper syncing at start
        bit_ctrs = [round((n+0.4) * SAMPLE_RATE/rate) for n in range(stream_length)]
        duration = 1000 / rate
        filter = get_bandpass(freq, SAMPLE_RATE, **kwargs)
        conv = np.convolve(filter, signal, mode='same')
        # conv = conv[len(filter)//2:]
        if kwargs.get('plot_conv'):
            plt.figure('conv for freq = {}'.format(str(freq)))
            plt.plot(conv)
            for i, bit_ctr in enumerate(bit_ctrs):
                plt.figure('conv for freq = {}'.format(str(freq)))
                if test_bit_stream[i]:
                    plt.axvline(x=bit_ctr, color='g')
                else:
                    plt.axvline(x=bit_ctr, color='r')
        for i, bit_ctr in enumerate(bit_ctrs):
            # TODO this is where QAM could come in, but sync issues
            print(conv[bit_ctr])
            conv_values[freq].append(np.sum(abs(conv[bit_ctr-3:bit_ctr+3])))
        plot_main_flag = False

    if kwargs.get('plot_conv') or kwargs.get('plot_main'):
        plt.draw()

    # Thresholding
    bit_streams = {freq: [] for freq in freqs}
    for freq in freqs:
        threshold = np.mean(conv_values[freq])
        for conv_value in conv_values[freq]:
            if conv_value > threshold:
                bit_streams[freq].append(1)
            else:
                bit_streams[freq].append(0)

    ret = []
    for stream in bit_streams.values():
        ret.extend(stream)
    return ret
