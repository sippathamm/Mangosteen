from scipy.io import wavfile

import numpy as np
import matplotlib.pyplot as plt


class FFT:
    def __init__(self, file_path):
        self.file_path = file_path
        self.Fs, self.y = wavfile.read(file_path)
        self.y = self.y / (2 ** 15)
        self.time = None
        self.frequency = None
        self.magnitude = None

    def analyze(self):
        self.magnitude = np.abs(np.fft.fft(self.y))
        self.frequency = np.abs(np.fft.fftfreq(len(self.magnitude), 1 / self.Fs))

    def get_max_frequency(self):
        max_index = np.argmax(self.magnitude)
        return self.frequency[max_index]

    def get_max_magnitude(self):
        return np.max(self.magnitude)

    def plot_fft(self):
        plt.plot(self.frequency[:len(self.frequency) // 2], self.magnitude[:len(self.magnitude) // 2])
        plt.title("FFT")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.show()
