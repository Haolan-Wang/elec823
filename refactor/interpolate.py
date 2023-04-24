import numpy as np
from scipy.interpolate import lagrange, interp1d

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def lagrange_interpolate(audiogram, audiogram_cfs, filter_cfs):
    # lagrange_poly = lagrange(audiogram_cfs, audiogram)
    lagrange_poly = interp1d(audiogram_cfs, audiogram, bounds_error=False, fill_value="extrapolate")
    return lagrange_poly(filter_cfs)

def get_central_frequencies(num_filter, lowfreq, highfreq):
    low_mel = hz_to_mel(lowfreq)
    high_mel = hz_to_mel(highfreq)

    mel_points = np.linspace(low_mel, high_mel, num_filter + 2)  # nfilt + 2 points to include bounds
    hz_points = mel_to_hz(mel_points)

    central_frequencies = hz_points[1:-1]  # exclude the first and last points
    return central_frequencies

def get_interpolated_audiogram(
    audiogram, 
    audiogram_cfs,
    num_filter=80, 
    lowfreq=0, 
    highfreq=8000, 
    ):
    """return: list of interpolated audiogram values"""
    interpolated_audiogram = []
    filter_cfs = get_central_frequencies(num_filter, lowfreq, highfreq)
    for i in range(len(audiogram)):
        interpolated_audiogram.append(lagrange_interpolate(audiogram[i], audiogram_cfs[i], filter_cfs))
    
    return interpolated_audiogram