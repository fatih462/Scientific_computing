import numpy as np

####################################################################################################
# Exercise 1: DFT

def dft_matrix(n: int) -> np.ndarray:
    """
    Construct DFT matrix of size n.

    Arguments:
    n: size of DFT matrix

    Return:
    F: DFT matrix of size n

    Forbidden:
    - numpy.fft.*
    """
    F = np.ones((n, n), dtype='complex128')

    omega = np.exp(-2*np.pi*1j/n)

    for i in range(n-1, 0, -1):
        for j in range(n-1, 0, -1):
            F[j, i] = omega**(i*j)

    F = 1/np.sqrt(n)*F


    return F


def is_unitary(matrix: np.ndarray) -> bool:
    """
    Check if the passed in matrix of size (n times n) is unitary.

    Arguments:
    matrix: the matrix which is checked

    Return:
    unitary: True if the matrix is unitary
    """

    konjugiert_transponiert = np.conjugate(matrix.T)

    ergebnis = np.dot(konjugiert_transponiert, matrix)

    rows, columns = np.shape(ergebnis)

    unitary = np.allclose(ergebnis, np.eye(rows, columns))

    return unitary


def create_harmonics(n: int = 128) -> (list, list):
    """
    Create delta impulse signals and perform the fourier transform on each signal.

    Arguments:
    n: the length of each signal

    Return:
    sigs: list of np.ndarrays that store the delta impulse signals
    fsigs: list of np.ndarrays with the fourier transforms of the signals
    """

    # list to store input signals to DFT
    sigs = []
    # Fourier-transformed signals
    fsigs = []

    F = dft_matrix(n)

    for i in range(n):
        delta = np.zeros(n)
        delta[i] = 1
        sigs.append(delta)
        fsigs.append(np.dot(F, delta))
    

    return sigs, fsigs


####################################################################################################
# Exercise 2: FFT

def shuffle_bit_reversed_order(data: np.ndarray) -> np.ndarray:
    
    length = len(data)
    binary_len = int(np.log2(length))

    #indizen = np.arange(length, dtype=np.uint8)
    shuffled_data = np.zeros(length, dtype='complex128')

    for i in range(length):
        binary_repr = np.binary_repr(i)
        binary_repr = binary_repr.zfill(binary_len)[::-1]
        new_element = int(binary_repr, 2)
        shuffled_data[new_element] = data[i]

    #shuffled_data = np.take(data, new_indizes)
    
    return shuffled_data


def fft(data: np.ndarray) -> np.ndarray:


    fdata = np.asarray(data, dtype='complex128')
    n = fdata.size

    # check if input length is power of two
    if not n > 0 or (n & (n - 1)) != 0:
        raise ValueError

    # first step of FFT: shuffle data
    fdata = shuffle_bit_reversed_order(data)

    # second step, recursively merge transforms
    for m in range(np.log2(n).astype(int)):
        for j in range(2**m):
            omega = np.exp(-2 * np.pi * 1j * j / (2**(m+1)))
            for k in range(n // (2**(m+1))):
                s = omega * fdata[j + (2*k + 1) * 2**m]
                save = fdata[j + 2*k * 2**m]
                fdata[j + 2*k * 2**m] = save + s
                fdata[j + (2*k + 1) * 2**m] = save - s


    # normalize fft signal
    fdata = fdata / np.sqrt(n)

    return fdata


def generate_tone(f: float = 261.626, num_samples: int = 44100) -> np.ndarray:
    """
    Generate tone of length 1s with frequency f (default mid C: f = 261.626 Hz) and return the signal.

    Arguments:
    f: frequency of the tone

    Return:
    data: the generated signal
    """

    # sampling range
    x_min = 0.0
    x_max = 1.0

    data = np.zeros(num_samples)

    # TODO: Generate sine wave with proper frequency
    #N = num_samples * x_max
    T = 1/(num_samples-1)
    omega = 2*np.pi*f
    t_seq = np.arange(num_samples) * T

    data = np.sin(omega*t_seq)

    return data


def low_pass_filter(adata: np.ndarray, bandlimit: int = 1000, sampling_rate: int = 44100) -> np.ndarray:
    
    #adata = np.array(adata)
    # translate bandlimit from Hz to dataindex according to sampling rate and data size
    bandlimit_index = int(bandlimit*adata.size/sampling_rate)

    # compute Fourier transform of input data
    fourier_data = np.fft.fft(adata)

    # set high frequencies above bandlimit to zero, make sure the almost symmetry of the transform is respected.
    for i in range(bandlimit_index + 1, fourier_data.size-bandlimit_index):
        fourier_data[i] = 0

    # compute inverse transform and extract real component
    adata_filtered = np.zeros(adata.shape[0])
    adata_filtered = np.fft.ifft(fourier_data).real

    return adata_filtered


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
