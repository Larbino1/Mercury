import sounddevice as sd
import soundfile as sf
import wave
import math
import time
import numpy as np
import struct
import matplotlib.pyplot as plt

SYNC_PULSE_WIDTH = 10
SYNC_FRAME_WIDTH = SYNC_PULSE_WIDTH*5
SYNC_PULSE_FREQ = 11000

SAMPLE_RATE = 44100

class AudioManager:
    def __init__(self):
        self.sd = sd
        self.sf = sf

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def playrec(self, file_to_play, file_to_save, duration, plot=True):
        frames = round((duration+1) * SAMPLE_RATE)
        data, fs = sf.read('rec/' + file_to_play)
        dataout = sd.playrec(data, fs, 1)
        time.sleep(duration+1)
        sf.write('rec/' + file_to_save, dataout, SAMPLE_RATE)
        if plot:
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


class OutputStream:
    def __init__(self, p, wvformat, channels, rate):
        self.p = p
        specs = p.get_default_output_device_info()
        print('Output specs: {}'.format(specs))
        self.format = wvformat
        self.channels = channels
        self.rate = rate

    def __enter__(self):
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            output=True,
        )
        return self.stream

    def __exit__(self, *args):
        self.stream.stop_stream()
        self.stream.close()


class InputStream:
    def __init__(self, p, wvformat, channels, rate, frames_per_buffer):
        self.p = p
        self.format = wvformat
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer

    def __enter__(self):
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            frames_per_buffer=self.frames_per_buffer,
            input=True,
        )
        return self.stream

    def __exit__(self, *args):
        self.stream.stop_stream()
        self.stream.close()


def plot_wav(filename):
    with wave.open('rec/' + filename, 'r') as f:
        # Extract Raw Audio from Wav File
        signal = f.readframes(-1)
        signal = np.frombuffer(signal, dtype='int16')

        plt.figure(1)
        plt.title('Signal Wave...')
        plt.plot(signal)
        plt.show()


def append_silence(audio, duration_milliseconds=500.0, rate = SAMPLE_RATE):
    """
    Adding silence is easy - we add zeros to the end of our array
    """
    num_samples = duration_milliseconds * (rate / 1000.0)

    for x in range(int(num_samples)):
        audio.append(0.0)

    return audio


def append_sinewave(audio, freq=150.0, duration_milliseconds=500.0, volume=1.0, rate=SAMPLE_RATE):
    num_samples = duration_milliseconds * (rate / 1000.0)

    for x in range(int(num_samples)):
        audio.append(volume * math.sin(2 * math.pi * freq * (x / SAMPLE_RATE)))

    return audio


def append_sinewaves(audio, freqs, duration_milliseconds=500.0, volume=1.0, rate = SAMPLE_RATE):
    num_samples = int(duration_milliseconds * (rate / 1000.0))
    appendage = [0]*num_samples
    for freq in freqs:
        sinewave = append_sinewave([], freq=freq, duration_milliseconds=duration_milliseconds)
        appendage = [a + b for a, b in zip(appendage, sinewave)]
    appendage = [a/8 for a in appendage]
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
    audio = append_sinewave(audio, duration_milliseconds=SYNC_PULSE_WIDTH, freq=SYNC_PULSE_FREQ)
    audio = append_silence(audio, duration_milliseconds=SYNC_PULSE_WIDTH)
    audio = append_sinewave(audio, duration_milliseconds=SYNC_PULSE_WIDTH, freq=SYNC_PULSE_FREQ)
    return audio


def encode(filename, bytes_array, freqs, rate=25, ):
    print('Encoding: {}'.format(bytes_array))
    audio = []
    audio = append_silence(audio, duration_milliseconds=500)

    total_bits = len(bytes_array)
    print('total_bits = {}'.format(total_bits))
    duration = 1000/rate

    audio.extend(get_sync_pulse())

    for i, byte in enumerate(bytes_array):
        active_freqs = []
        bits = np.unpackbits(byte)
        for freq, bit in zip(freqs, bits):
            if bit:
                active_freqs.append(freq)
        # print('{}/{}'.format(i, total_bits))
        audio = append_sinewaves(audio, freqs=active_freqs, duration_milliseconds=duration)

    audio = append_silence(audio, duration_milliseconds=500)

    save_wav(audio, filename)
    print('DONE')


def decode(filename, rate, byte_count, freqs):
    print('Decoding')
    N = byte_count
    with wave.open('rec/' + filename) as f:
        signal = f.readframes(-1)
        signal = np.frombuffer(signal, dtype='int16')
        width = SAMPLE_RATE/rate
        print(width)

        # Find start

        audio = get_sync_pulse()
        sig = signal[:SAMPLE_RATE*2]
        conv = np.convolve(audio, sig)
        i_best = np.argmax(conv)

        plt.figure('sync')
        plt.plot(sig / max(sig), color='r')
        plt.plot(conv / max(conv))
        plt.axvline(x=i_best, color='g')

        signal = signal[i_best:]

        A_list = []
        for n in range(N):
            a = round(n*width)
            b = round((n+1)*width)
            # plt.figure('main')
            # plt.axvline(x=a, color='r')
            ft = abs(np.fft.fft(signal[a:b]))
            x = np.linspace(0, SAMPLE_RATE, len(ft))
            fft_indices = [round(freq * len(ft) / SAMPLE_RATE) for freq in freqs]
            # plt.figure(n)
            # plt.plot(x, ft)
            # for index in fft_indices:
                # plt.axvline(x=index*SAMPLE_RATE/len(ft), color='r')
            A_list.extend([ft[index] for index in fft_indices])

        print(A_list)
        threshold = np.mean(A_list)
        print('threshold = {}'.format(threshold))

        # for n in range(N):
        #     plt.figure(n)
        #     plt.axhline(y=threshold)
        #plt.show()

        ret = []
        for area in A_list:
            if area > threshold:
                ret.append(1)
            else:
                ret.append(0)
        # print(ret)
        return ret

