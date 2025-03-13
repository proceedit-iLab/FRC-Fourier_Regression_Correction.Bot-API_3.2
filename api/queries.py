import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2.extensions import AsIs
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import json
import requests
import traceback
from scipy.fftpack import fft, ifft, fftfreq
import seaborn as sns
import sys

from .utils import *

def resolve_fitApplyFourierCorrection(_, info, actual, predicted_targets, positions_day_number):
    try:
        correction_df = pd.DataFrame({'actual': actual, 'predicted_targets':predicted_targets})
        correction_df['predicted_actual_targets_ratio'] = correction_df['predicted_targets'] / correction_df['actual']
        
        window_size = 10  
        top_n = 80
        fft_features_list = [None] * window_size
        reconstructed_signals = []
        predicted_values = []

        for i in range(window_size, len(actual)):  
            window_data = correction_df['predicted_actual_targets_ratio'].iloc[i - window_size:i].values

            features, fft_values, top_indices = extract_fft_features(window_data, top_n_percent = top_n)
            fft_features_list.append(features)  

            next_value = reconstruct_signal_from_fft(fft_values, top_indices, n_future=1)[0]
            predicted_values.append(next_value)

        fft_features_df = pd.json_normalize(fft_features_list) # Convert FFT features into a DataFrame

        # correction_df = pd.concat([correction_df.reset_index(drop=True), fft_features_df.reset_index(drop=True)], axis=1)
        correction_df['predicted_next_value'] = [0] * window_size + predicted_values

        print(correction_df.tail())

        correction_df['corrected_targets'] = correction_df['predicted_next_value'] * correction_df['predicted_targets']
        
        correction_factors = correction_df['predicted_next_value'].values.tolist()
        corrected_targets = correction_df['corrected_targets'].values.tolist()

    except Exception as error:
        print(error,traceback.format_exc())
        response = {
            'success': False,
            'error': traceback.format_exc()
        }
        return response

    response = {
        'success':True,
        'error': None,
        'correction_factors': correction_factors,
        'corrected_targets': corrected_targets
    }
    
    return response