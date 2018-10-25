import matplotlib.pyplot as plt
import numpy as np
import wave
import time
from threading import Event, Thread

import utils
import tests

am = utils.AudioManager()


def play_and_record(file_to_play, file_to_save):
    duration = utils.get_wav_duration(file_to_play)

    play = Thread(target=am.spk.play_wav, args=(file_to_play,))
    record = Thread(target=am.mic.record, args=(file_to_save, duration + 1.5))
    record.start()
    time.sleep(0.75)
    play.start()
    play.join()
    record.join()


rate = 150
base_freq = 1000
freqs = [base_freq + 1000*i for i in range(8)]

print('Frequencies: {}'.format(freqs))

utils.encode('bin.wav', tests.testbytes, freqs, rate)

play_and_record('bin.wav', '_bin.wav')
ans = utils.decode('_bin.wav', rate, len(tests.testbytes), freqs)

print(ans)
print(list(tests.testbits))

e = 0
for i in zip(ans, list(tests.testbits)):
    if i[0] != i[1]:
        e += 1
print('{}% error'.format(round(100*e/len(tests.testbits))))

if ans == list(tests.testbits):
    print("YEET!")
else:
    print("SHITE")
