import numpy as np


data = [10, 20, 30, 40, 50, 60, 70]
data2 = [8, 5, 9, 13, 4, 6.6, 3]
data = np.array(data)
data2 = np.array(data2)

data_fft = np.fft.fft(data)
data2_fft = np.fft.fft(data2)


cross_corr_time = np.multiply(data.T, data2)

# Kreuzkorrelation im Frequenzbereich
data_fft = np.fft.fft(data)
data2_fft = np.fft.fft(data2)
cross_corr_freq = np.multiply(data_fft.T, data2_fft)

print(data_fft)
print("pause")
print(data2_fft)
print("pause")
print("Kreuzkorrelation im Zeitbereich:", cross_corr_time)
print("Kreuzkorrelation im Frequenzbereich:", cross_corr_freq)
