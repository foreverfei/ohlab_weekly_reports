import scipy as sci
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import stft

# class  ZeroError(Exception):
#     def __init__(self, error ,index):
#         self.error = error
#         self.index = index
#
#     def __str__(self):
#
#         return 'spectrum出现1' + str(spectrum[self.index])

if __name__ == '__main__':

    sound_rxd = sci.io.wavfile.read('_4.wav')

    sample_rate, sound = sound_rxd
    frequencies, t, spectrum = stft(sound , sample_rate  , nperseg=256)
    # if 0 in abs(spectrum) :
    #
    #     indice = np.where(spectrum == 0)
    #     raise ZeroError(spectrum ,indice )
    #
    #     np.delete(indice, indice)


    spectrum = np.log(abs(spectrum) + 0.000001) #防止log 0报错


    plt.pcolormesh(t, frequencies , spectrum ,shading = 'gouraud' )
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()