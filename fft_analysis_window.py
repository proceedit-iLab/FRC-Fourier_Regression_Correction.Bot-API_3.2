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

pd.set_option('display.max_columns', None)

headers = {
    'Accept-Encoding': 'gzip, deflate, br',
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Connection': 'keep-alive',
    'DNT': '1'
}

def fetch_features_targets_input_data(asset,dataset_start_date, connection_params):
    error = None  # Initialize error variable outside of try block
    try:
        with psycopg2.connect(**connection_params) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur: 
                print('You are connect to the Database:', connection_params['dbname'])
                current_financial_asset_symbol = [asset]
                for n in current_financial_asset_symbol:   
                    # check_script = ''' SELECT * FROM information_schema.views
                    #              WHERE table_name = 'features_targets_input_view'  
                    #              AND table_schema = 'ASSET_%s' '''
                    check_script = ''' SELECT * FROM information_schema.views
                                 WHERE table_name = 'features_targets_input_view'  
                                 AND table_schema = 'ASSET_%s' '''
                    val = AsIs(current_financial_asset_symbol[current_financial_asset_symbol.index(n)])
                    cur.execute(check_script, (val,))
                    bool(cur.rowcount)
                    if cur.rowcount == 1: 
                        # select_script = ('''SELECT * FROM "ASSET_%s".features_targets_input_view WHERE features_targets_input_view."cleaned_raw_features_environment_PK" = 8''')
                        select_script = ('''SELECT * FROM "ASSET_%s".features_targets_input_data_view WHERE features_targets_input_data_view."environment_PK" = 8''')
                        val = AsIs(current_financial_asset_symbol[current_financial_asset_symbol.index(n)])
                        cur.execute(select_script, (val,))
                        data = cur.fetchall()
                        cols = []
                        for elt in cur.description:
                            cols.append(elt[0])
                        features_targets_input_data = pd.DataFrame(data=data, columns=cols)
                        #line 48 added for AAPL
                        features_targets_input_data['DCP_date_current_period'] = features_targets_input_data['DCP_date_current_period'].astype(str)
                        features_targets_input_data=features_targets_input_data.loc[features_targets_input_data['DCP_date_current_period'] >= dataset_start_date]
                        # features_targets_input_data=features_targets_input_data.loc[features_targets_input_data['DCP_date_current_period'] >= dataset_start_date]                    
                        error = None  # Reset error if no exception occurs
                    else:
                        # print('Table in schema', val, 'not found')
                        features_targets_input_data = None
                        error = 'Table in schema not found'
    except Exception as e:
        error = f'An error occurred: {e}' # Update error if an exception occurs
        traceback.print_exc()
        features_targets_input_data = None
    finally:
        if 'conn' in locals() and conn is not None:  # Check if conn is defined before closing
            conn.close()
    features_targets_input_data.reset_index(drop=True,inplace=True)
    return features_targets_input_data, error

DBNAME='dyDATA_new'
DATABASE_HOST='database-1.ctzm0hf7fhri.eu-central-1.rds.amazonaws.com'
USER='postgres'
PASSWORD='Proc2023awsrdspostgresql'
DATABASE_PORT=5432

conn = psycopg2.connect(
    dbname=DBNAME,
    user=USER,
    password=PASSWORD,
    host=DATABASE_HOST,
    port=DATABASE_PORT
)

connection_params = {
    "host": DATABASE_HOST,
    "port": DATABASE_PORT,
    "user": USER,
    "password": PASSWORD,
    "dbname": DBNAME
}

dataset_start_date = '2020-01-01' #for TSLA
case = 'TSLA'

# features_targets_input_data,error = fetch_features_targets_input_data(case,dataset_start_date,connection_params)
# dohlcav_mpnxp_data = features_targets_input_data.copy()
dohlcav_mpnxp_data = pd.read_csv('/lrn/fft-analysis/lmx_predictions.csv')
# print(dohlcav_mpnxp_data.tail())
# dohlcav_mpnxp_data= dohlcav_mpnxp_data.iloc[::-1].reset_index(drop=True)
# print(dohlcav_mpnxp_data.head())
# sys.exit()

df = pd.DataFrame()

# Compute daily price change (Delta)
df['ACPCP_adjusted_close_price_current_period'] = dohlcav_mpnxp_data['ACPCP_adjusted_close_price_current_period']
# df['delta'] = dohlcav_mpnxp_data['ACPCP_adjusted_close_price_current_period'].diff().fillna(0)
df['delta'] = dohlcav_mpnxp_data['ACPCP_adjusted_close_price_current_period']

price_mean = df['ACPCP_adjusted_close_price_current_period'].mean() # exp.1 working with mean

def reconstruct_signal_from_fft(original_fft, top_indices):
    # Create a new FFT array with zeroed-out frequencies except for the selected ones
    filtered_fft = np.zeros_like(original_fft, dtype=complex)
    filtered_fft[top_indices] = original_fft[top_indices]  # Keep only top frequencies
    
    # Use inverse FFT to reconstruct the signal
    return np.real(ifft(filtered_fft))

def extract_fft_features(prices, top_n_percent=20):

    n = len(prices) 
    fft_values = fft(prices)
    freqs = fftfreq(n)
    
    amplitudes = np.abs(fft_values)[:n//2] 
    phases = np.angle(fft_values)[:n//2] 

    sorted_indices = np.argsort(amplitudes)[::-1]  # sorting in desc order
    top_n = max(1, int(len(sorted_indices) * (top_n_percent / 100)))

    # selecting top frequencies
    top_freqs = freqs[sorted_indices][:top_n]
    top_amplitudes = amplitudes[sorted_indices][:top_n]
    top_phases = phases[sorted_indices][:top_n]

    # additional features
    total_energy = np.sum(amplitudes ** 2)
    top_energy = np.sum(top_amplitudes ** 2)
    energy_ratio = top_energy / total_energy if total_energy > 0 else 0

    # Extract main features
    features = {
        'dominant_frequency': top_freqs[0],  
        'sum_top_amplitudes': np.sum(top_amplitudes),
        'energy_ratio': energy_ratio
    }

    for i in range(min(3, top_n)): 
        features[f'top_freq_{i+1}'] = top_freqs[i]
        features[f'top_amp_{i+1}'] = top_amplitudes[i]
        features[f'top_phase_{i+1}'] = top_phases[i]

    return features

window_size = 30 
fft_features_list = [None] * window_size
reconstructed_signals = []

for i in range(window_size, len(df)):                         # rolling window to extract fft info
    window_data = df['delta'].iloc[i - window_size:i].values
    features = extract_fft_features(window_data)
    fft_features_list.append(features)

fft_features_df = pd.json_normalize(fft_features_list)

# nan_padding = pd.DataFrame(np.nan, index=range(window_size), columns=fft_features_df.columns)
# fft_features_df = pd.concat([nan_padding, fft_features_df], ignore_index=True)

# print(len(fft_features_df)) 
# print(len(df))
# sys.exit()
df = pd.concat([df.reset_index(drop=True), fft_features_df.reset_index(drop=True)], axis=1)

print(df.tail())


