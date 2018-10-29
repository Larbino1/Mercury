import sounddevice as sd
import soundfile as sf
import wave
import math
import time
import numpy as np
import struct
import matplotlib.pyplot as plt

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
        time.sleep(t)
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
    num_samples = round(duration_milliseconds * (rate / 1000.0))
    audio = []
    for x in range(num_samples):
        audio.append(volume * math.sin(2 * math.pi * freq * (x / SAMPLE_RATE)))
    return audio

def get_silence(duration_milliseconds, rate=SAMPLE_RATE):
    num_samples = round(duration_milliseconds * (rate / 1000.0))
    audio = []
    for x in range(int(num_samples)):
        audio.append(0.0)
    return audio


def append_sinewaves(audio, freqs, duration_milliseconds=500.0, volume=1.0, rate=SAMPLE_RATE):
    num_samples = round(duration_milliseconds * (rate / 1000.0))
    appendage = [0]*num_samples
    if freqs:
        for freq in freqs:
            sinewave = get_sinewave(freq, duration_milliseconds)
            appendage = [a + b for a, b in zip(appendage, sinewave)]
        appendage = [a/len(freqs) for a in appendage]
    audio.extend(appendage)
    return audio


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
            wav_file.writeframes(struct.pack('h', int( sample * 32767.0 )))


def get_wav_duration(filename):
    with wave.open('rec/' + filename, 'r') as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration


def get_sync_pulse():
    audio = []
    audio = append_sinewave(audio, duration_milliseconds=SYNC_PULSE_WIDTH, freq=SYNC_PULSE_FREQ)
    audio = append_silence(audio, duration_milliseconds=SYNC_PULSE_WIDTH)
    print(audio)
    audio = append_sinewave(audio, duration_milliseconds=SYNC_PULSE_WIDTH, freq=SYNC_PULSE_FREQ)
    audio = append_silence(audio, duration_milliseconds=SYNC_PULSE_WIDTH)
    audio = append_sinewave(audio, duration_milliseconds=SYNC_PULSE_WIDTH, freq=SYNC_PULSE_FREQ)
    return audio


def chunk(l, n):
    """Yield successive n-sized chunks from l."""
    if len(l)%n !=0:
        raise Exception('Number of bits must be divisible by {}'.format(n))
    for i in range(0, len(l), n):
        yield l[i:i + n]


def calc_error(correct_data, recieved_data):
    e = 0
    for i in zip(correct_data, recieved_data):
        if i[0] != i[1]:
            e += 1
    pcnt_error = round(100*e/len(correct_data), 3)
    print('{}% error'.format(pcnt_error))
    return pcnt_error


def plot_smooth_error_graph(correct_data, recieved_data):
    plt.figure('errors')
    errors = np.bitwise_xor(recieved_data, correct_data)
    sigma = 5
    x = np.linspace(-3*sigma, 3*sigma, 100*sigma)
    gaussian = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * sigma**2))
    smooth = np.convolve(gaussian, errors)
    plt.plot(smooth)
    plt.show()


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
    assert np.sum(B) == no_of_bits
    return B


def split_data_into_streams(bit_array, data_rates):
    B_cum = get_split_stream_cumulative(len(bit_array), data_rates)
    ret = []
    for n in range(len(B_cum)-1):
        ret.append(bit_array[B_cum[n]:B_cum[n+1]])
    return np.array(ret)


