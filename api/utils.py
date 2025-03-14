import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2.extensions import AsIs
import json
import requests
import sys
import traceback
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq
import seaborn as sns
from scipy.stats import linregress

def extract_fft_features(signal, top_n_percent=80):

    n = len(signal) 
    fft_values = fft(signal)
    freqs = fftfreq(n)
    print('Total frequencies : ',len(fft_values))
    # sys.exit()
    
    amplitudes = np.abs(fft_values) #[:n//2] 
    phases = np.angle(fft_values) #[:n//2] 

    sorted_indices = np.argsort(amplitudes)[::-1]  # sorting in desc order
    top_n = max(1, int(len(sorted_indices) * (top_n_percent / 100)))

    # selecting top frequencies
    top_freqs = freqs[sorted_indices][:top_n]
    top_amplitudes = amplitudes[sorted_indices][:top_n]
    top_phases = phases[sorted_indices][:top_n]
    print('Number of dominant frequencies : ',len(top_freqs))

    # additional features
    total_energy = np.sum(amplitudes ** 2)
    top_energy = np.sum(top_amplitudes ** 2)
    energy_ratio = top_energy / total_energy if total_energy > 0 else 0

    # main features
    features = {
        'dominant_frequency': top_freqs[0],  
        'sum_top_amplitudes': np.sum(top_amplitudes),
        'energy_ratio': energy_ratio
    }

    for i in range(min(10, top_n)): 
        features[f'top_freq_{i+1}'] = top_freqs[i]
        features[f'top_amp_{i+1}'] = top_amplitudes[i]
        features[f'top_phase_{i+1}'] = top_phases[i]

    return features, fft_values, sorted_indices[:top_n]  # also returns all fft values with top indices

# def reconstruct_signal_from_fft(original_fft, top_indices, n_future=1):
#     n = len(original_fft)  
#     t = np.arange(n + n_future) # extend time for predicting

#     reconstructed_signal = np.zeros_like(t, dtype=float)
#     for index in top_indices:
#         freq = index / n
#         amplitude = np.abs(original_fft[index]) / n
#         phase = np.angle(original_fft[index])

#         reconstructed_signal += amplitude * np.cos(2 * np.pi * freq * t + phase) # x(t)=Acos(2πft+ϕ)

#     return reconstructed_signal[-n_future:] 

def reconstruct_signal_from_fft(original_fft, top_indices, n_future=1):
    n = len(original_fft)
    extended_fft = np.zeros(n + n_future, dtype=complex)
    extended_fft[:n] = original_fft  # Copy original FFT coefficients

    # Reconstruct the signal using IFFT
    reconstructed_signal = np.fft.ifft(extended_fft)
    # print("reconstructed signal : ", reconstructed_signal)
    # print("len reconstructed signal : ",len(reconstructed_signal))

    return reconstructed_signal[-n_future:]  # Return extrapolated values