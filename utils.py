import sounddevice as sd
import soundfile as sf
import wave
import math
import time
import numpy as np
import struct
import matplotlib.pyplot as plt
import scipy.signal as signal

from consts import *


class AudioManager:
    def __init__(self):
        self.sd = sd
        self.sf = sf

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def playrec(self, file_to_play, file_to_save, **kwargs):
        duration = get_wav_duration(file_to_play)
        frames = round((duration+1) * SAMPLE_RATE)
        data, fs = sf.read('rec/' + file_to_play)
        dataout = sd.playrec(data, fs, 1)
        time.sleep(duration+1)
        sf.write('rec/' + file_to_save, dataout, SAMPLE_RATE)
        if kwargs.get('plot_ideal_signal'):
            print("Showing plot of recorded data")
            print(data)
            plt.plot(data)
            plt.show()

    def record(self, filename, t, plot=False):
        print("Recording to {} for {} seconds".format(filename, t))
        frames = round(t * SAMPLE_RATE)
        data = sd.rec(frames, SAMPLE_RATE, channels=1)
        time.sleep(t+1)
        sf.write('rec/' + filename, data, SAMPLE_RATE)
        if plot:
            print("Showing plot of recorded data")
            print(data)
            plt.plot(data)
            plt.show()

    def play_wav(self, filename):
        data, fs = sf.read('rec/' + filename)
        sd.play(data, fs)

    def test_tone(self):
        self.play_wav('test.wav')


def plot_wav(filename):
    with wave.open('rec/' + filename, 'r') as f:
        # Extract Raw Audio from Wav File
        signal = f.readframes(-1)
        signal = np.frombuffer(signal, dtype='int16')

        plt.figure(1)
        plt.title('Signal Wave...')
        plt.plot(signal)
        plt.show()


def append_silence(audio, duration_milliseconds=500.0, rate=SAMPLE_RATE):
    audio.extend(get_silence(duration_milliseconds))
    return audio


def append_sinewave(audio, freq, duration_milliseconds, volume=1.0, rate=SAMPLE_RATE):
    audio.extend(get_sinewave(freq, duration_milliseconds))
    return audio


def get_sinewave(freq, duration_milliseconds, volume=1.0, rate=SAMPLE_RATE):
    num_samples = int(duration_milliseconds * (rate / 1000.0))
    audio = []
    for x in range(num_samples):
        audio.append(volume * math.sin(2 * math.pi * freq * (x / SAMPLE_RATE)))
    return audio


def get_neg_sinewave(freq, duration_milliseconds, volume=1.0, rate=SAMPLE_RATE):
    return [-i for i in get_sinewave(freq, duration_milliseconds, volume, rate)]


def get_coswave(freq, duration_milliseconds, volume=1.0, rate=SAMPLE_RATE):
    num_samples = int(duration_milliseconds * (rate / 1000.0))
    audio = []
    for x in range(num_samples):
        audio.append(volume * math.cos(2 * math.pi * freq * (x / SAMPLE_RATE)))
    return audio


def get_neg_coswave(freq, duration_milliseconds, volume=1.0, rate=SAMPLE_RATE):
    return [-i for i in get_coswave(freq, duration_milliseconds, volume, rate)]


def get_silence(duration_milliseconds, rate=SAMPLE_RATE):
    num_samples = int(duration_milliseconds * (rate / 1000.0))
    audio = []
    for x in range(int(num_samples)):
        audio.append(0.0)
    return audio


def append_sinewaves(audio, freqs, duration_milliseconds=500.0, volume=1.0, rate=SAMPLE_RATE):
    num_samples = int(duration_milliseconds * (rate / 1000.0))
    appendage = [0]*num_samples
    if freqs:
        for freq in freqs:
            sinewave = get_sinewave(freq, duration_milliseconds)
            appendage = [a + b for a, b in zip(appendage, sinewave)]
        appendage = [a/len(freqs) for a in appendage]
    audio.extend(appendage)
    return audio


def get_bandpass(freq, sample_rate, half_width=None, **kwargs):
    if not half_width:
        half_width = freq/20
    fL = (freq - half_width)/sample_rate  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    fH = (freq + half_width)/sample_rate  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    b = half_width/sample_rate  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)
    x1 = 2 * fH * (n - (N - 1) / 2.)

    # Compute a low-pass filter with cutoff frequency fH.
    hlpf = np.sinc(x1)
    hlpf *= np.blackman(N)
    hlpf = hlpf / np.sum(hlpf)

    # Compute a high-pass filter with cutoff frequency fL.
    x2 = 2 * fL * (n - (N - 1) / 2.)
    hhpf = np.sinc(x2)
    hhpf *= np.blackman(N)
    hhpf = hhpf / np.sum(hhpf)
    hhpf = -hhpf
    centre = (N - 1) // 2
    hhpf[centre] += 1

    # Convolve both filters.
    h = np.convolve(hlpf, hhpf)
    if kwargs.get('plot_filters'):
        plt.figure('Filters')
        plt.plot(h)
        plt.draw()

        mag_plot(h)
    return h


def save_wav(audio, file_name):
    # Open up a wav file
    with wave.open('rec/' + file_name, "w") as wav_file:

        # wav params
        nchannels = 1

        sampwidth = 2

        # 44100 is the industry standard sample rate - CD quality.  If you need to
        # save on file size you can adjust it downwards. The stanard for low quality
        # is 8000 or 8kHz.
        nframes = len(audio)
        comptype = "NONE"
        compname = "not compressed"
        wav_file.setparams((nchannels, sampwidth, SAMPLE_RATE, nframes, comptype, compname))

        # WAV files here are using short, 16 bit, signed integers for the
        # sample size.  So we multiply the floating point data we have by 32767, the
        # maximum value for a short integer.  NOTE: It is theortically possible to
        # use the floating point -1.0 to 1.0 data directly in a WAV file but not
        # obvious how to do that using the wave module in python.
        for sample in audio:
            wav_file.writeframes(struct.pack('h', int(sample * 32767.0)))


def get_wav_duration(filename):
    with wave.open('rec/' + filename, 'r') as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration


def get_sync_pulse():
    sync_pulse = get_sync_pulse1()
    # sync_pulse *= np.blackman(len(sync_pulse))
    return sync_pulse


def get_sync_pulse1():
    audio = []
    audio = append_sinewave(audio, duration_milliseconds=SYNC_PULSE_PIP_WIDTH, freq=SYNC_PULSE_FREQ)
    audio = append_silence(audio, duration_milliseconds=SYNC_PULSE_PIP_WIDTH)
    audio = append_sinewave(audio, duration_milliseconds=SYNC_PULSE_PIP_WIDTH, freq=SYNC_PULSE_FREQ)
    audio = append_silence(audio, duration_milliseconds=SYNC_PULSE_PIP_WIDTH)
    audio = append_sinewave(audio, duration_milliseconds=SYNC_PULSE_PIP_WIDTH, freq=SYNC_PULSE_FREQ)
    return audio


def get_sync_pulse2():
    num_samples = int(10 * (SAMPLE_RATE / 1000.0))
    audio = []
    volume = 1
    X = np.linspace(-5*np.pi, 5*np.pi, num_samples)
    for x in X:
        audio.append(np.sinc(x))
    audio = [(1-abs(i)) * np.sign(i) for i in audio]
    plt.figure('syncpulse')
    plt.plot(audio)
    plt.draw()
    return audio


def get_data_rates(freqs):
    return [freq/10 for freq in freqs]


def get_symbol_rates(freqs):
    symbol_widths = [len(get_bandpass(freq)) for freq in freqs]
    symbol_rates = [SAMPLE_RATE/symbol_width for symbol_width in symbol_widths]
    return symbol_rates


def pad(bit_list, n):
    data_bit_count = int(np.ceil(np.log2(n)))
    # print("no of data bits = {}".format(data_bit_count))
    no_of_padding_bits = (n - (len(bit_list) + data_bit_count) % n) % n
    # print("no of padding bits = {}".format(no_of_padding_bits))
    format_str = "{:0>" + str(data_bit_count) + "b}"
    padding_bits = format_str.format(no_of_padding_bits)
    # print("chunking {} bits into {}s".format(len(l), n))
    # print("Adding data bits {} ".format(padding_bits))
    # print(padding_bit_count)
    for i in padding_bits[::-1]:
        bit_list = np.insert(bit_list, 0, int(i))
    for i in range(no_of_padding_bits):
        bit_list = np.append(bit_list, 0)
    return bit_list


def unpad(bit_list, n):
    data_bit_count = int(np.ceil(np.log2(n)))
    data_bits = bit_list[:data_bit_count]
    no_of_padding_bits = 0
    for bit in data_bits:
        no_of_padding_bits = (no_of_padding_bits << 1) | bit  # base 2 notation
    # print("Bit list in unpad {}".format(bit_list))
    # print("Data bit count: {}".format(data_bit_count))
    # print("No of padding bits: {}".format(no_of_padding_bits))
    if no_of_padding_bits == 0:
        bit_list = bit_list[data_bit_count:]
    else:
        bit_list = bit_list[data_bit_count:-no_of_padding_bits]
    # print("Unpadded: {}".format(bit_list))
    return bit_list


def chunk(l, n):
    """Yield successive n-sized chunks from l."""
    if len(l)%n != 0:
        raise Exception('Number of bits must be divisible by {}'.format(n))
    for i in range(0, len(l), n):
        yield l[i:i + n]


def calc_error(correct_data, recieved_data):
    e = 0
    for i in zip(correct_data, recieved_data):
        if i[0] != i[1]:
            e += 1
    pcnt_error = round(100*e/len(correct_data), 3)
    return pcnt_error


def calc_error_per_freq(sent_data, recieved_data, freqs):
    bit_rates = get_data_rates(freqs)
    sent_streams = split_data_into_streams(sent_data, bit_rates)
    recieved_streams = split_data_into_streams(recieved_data, bit_rates)
    for freq, sent_dta, recieved_dta in zip(freqs, sent_streams, recieved_streams):
        error = calc_error(sent_dta, recieved_dta)
        print('Freq:{:8}, {}% error'.format(freq, error))
    error = calc_error(sent_data, recieved_data)
    print('TOTAL {}% error'.format(error))


def plot_smooth_error_graph(correct_data, recieved_data):
    plt.figure('errors')
    errors = np.bitwise_xor(recieved_data, correct_data)
    sigma = 1
    x = np.linspace(-3*sigma, 3*sigma, 100*sigma)
    gaussian = (2*np.pi)/(sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * sigma**2))
    smooth = np.convolve(gaussian, errors)
    plt.plot(smooth)
    plt.plot(10*errors)
    plt.axis((0, len(correct_data), 0, 60))
    plt.draw()
    if len(correct_data) != len(recieved_data):
        print("RECIEVED DIFFERENT LENGTH OF DATA")


def assert_arrays_equal(array1, array2):
    for a, b in zip(array1, array2):
        assert a == b, 'Arrays not equal!'


def check_given_rates(data_rates, N):
    if type(data_rates) is int or type(data_rates) is float:
        # Check for only one data rate being given and build list
        print("Assuming uniform data rate per frequency, expected list got '{}'".format(data_rates))
        rate = data_rates
        data_rates = [rate/N for _i in range(N)]
    return data_rates


def get_split_stream_cumulative(no_of_bits, bit_rates):
    B = get_split_stream_lengths(no_of_bits, bit_rates)
    B_cum = [sum(B[:i + 1]) for i in range(len(B))]
    assert B_cum[-1] == no_of_bits
    B_cum.insert(0, 0)
    return B_cum


def get_split_stream_lengths(no_of_bits, bit_rates):
    total_data_rate = sum(bit_rates)
    total_bits = no_of_bits
    B = [round(dr * total_bits / total_data_rate) for dr in bit_rates]
    if np.sum(B) != no_of_bits:
        B[-1] = total_bits - np.sum(B[:-1])
    assert np.sum(B) == no_of_bits
    return B


def split_data_into_streams(bit_array, data_rates):
    B_cum = get_split_stream_cumulative(len(bit_array), data_rates)
    ret = []
    for n in range(len(B_cum)-1):
        ret.append(bit_array[B_cum[n]:B_cum[n+1]])
    return np.array(ret)


#Magnitude plot
def mag_plot(b,a=1):
    w,h = signal.freqz(b,a)
    h_dB = abs(h)
    plt.figure('Freq response of filters')
    plt.plot(w,h_dB)
    plt.ylabel("Magnitude")
    plt.xlabel('Normalized Frequency')
    plt.title('Frequency response')


def bits2string(bits):
    return ''.join(str(i) for i in bits)
