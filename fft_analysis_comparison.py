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

pd.set_option('display.max_columns', None)

headers = {
    'Accept-Encoding': 'gzip, deflate, br',
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Connection': 'keep-alive',
    'DNT': '1'
}

def sir_parameters(x,y): #sir stands for slope, intercept, rvalue (actually there's also the average trend line distance or avg_tld, but it came later)
  x=np.array(x)
  y=np.array(y)
  analytical_params = linregress(x, y)

  slope = analytical_params.slope
  intercept = analytical_params.intercept
  rvalue = analytical_params.rvalue #pay attention that here we have the correlaton coefficient (so not r2 that is the coefficient of determination)

  y_trend_line = slope*x + intercept #this is computed just for the avg_tld
  avg_trend_line_distance = np.mean(np.abs(y_trend_line - y)/y_trend_line)

  return slope, intercept, rvalue**2, avg_trend_line_distance

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
# # print(dohlcav_mpnxp_data.tail())
dohlcav_mpnxp_data = pd.read_csv('/lrn/fft-analysis/predicted_corrected_targets.csv')
# dohlcav_mpnxp_data= dohlcav_mpnxp_data.iloc[::-1].reset_index(drop=True)
# print(dohlcav_mpnxp_data.head())
# sys.exit()

df = pd.DataFrame()

# Compute daily price change (Delta)
# df['ACPCP_adjusted_close_price_current_period'] = dohlcav_mpnxp_data['ACPCP_adjusted_close_price_current_period']
# df['delta'] = dohlcav_mpnxp_data['ACPCP_adjusted_close_price_current_period'].diff().fillna(0)
# df['delta'] = dohlcav_mpnxp_data['ACPCP_adjusted_close_price_current_period']
df['delta'] = dohlcav_mpnxp_data['raw predicted'] / dohlcav_mpnxp_data['actual'] 

price_mean = df['delta'].mean() # exp.1 working with mean

def extract_fft_features(prices, top_n_percent=20):
    """
    Extracts FFT features from a given Adjusted Price time series.
    
    Parameters:
        prices (pd.Series): Adjusted Price time series
        top_n_percent (int): Percentage of top frequencies to keep

    Returns:
        dict: Extracted features (amplitudes, phases, dominant frequency, energy ratio)
    """
    n = len(prices) 
    fft_values = fft(prices)
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

    # Extract main features
    features = {
        'dominant_frequency': top_freqs[0],  
        'sum_top_amplitudes': np.sum(top_amplitudes),
        'energy_ratio': energy_ratio
    }

    for i in range(min(30, top_n)): 
        features[f'top_freq_{i+1}'] = top_freqs[i]
        features[f'top_amp_{i+1}'] = top_amplitudes[i]
        features[f'top_phase_{i+1}'] = top_phases[i]

    return features, fft_values, sorted_indices[:top_n]  # Also return full FFT and top indices

# Function to reconstruct signal using inverse FFT
def reconstruct_signal_from_fft(original_fft, top_indices):
    filtered_fft = np.zeros_like(original_fft, dtype=complex)
    filtered_fft[top_indices] = original_fft[top_indices]  # Keep only top frequencies
    
    return np.real(ifft(filtered_fft))

window_size = 10  
fft_features_list = [None] * window_size
reconstructed_signals = []
predicted_values = []

for i in range(window_size, len(df)):  
    window_data = df['delta'].iloc[i - window_size:i].values

    features, fft_values, top_indices = extract_fft_features(window_data)
    fft_features_list.append(features)

    reconstructed_signal = reconstruct_signal_from_fft(fft_values, top_indices)
    reconstructed_signals.append(reconstructed_signal)

    if i == 10:
        trend_slope, trend_intercept, trend_r2, dispersion = sir_parameters(window_data, reconstructed_signal)
        
        x_index = np.arange(len(window_data))
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x=x_index, y=window_data, label="Actual", ax=ax)
        sns.lineplot(x=x_index, y=reconstructed_signal, label="Reconstructed", ax=ax)

        # Set plot labels and title
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel('Error', fontsize=12)
        ax.set_title(f'Comparison of Original vs Reconstructed', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)

        # Add calculated statistics to the plot
        text_x = 1.02  # position outside the right edge of the plot
        text_y = 0.9   # starting vertical position
        text_spacing = 0.07  # space between text lines
        stats = [
            f"Trend Slope: {trend_slope:.4f}",
            f"Trend Intercept: {trend_intercept:.4f}",
            f"Trend R²: {trend_r2:.4f}",
            f"Standard Deviation: {np.std(window_data):.4f}",
            f"Dispersion: {dispersion:.4f}",
        ]
        for j, stat in enumerate(stats):
            plt.figtext(text_x, text_y - j * text_spacing, stat, fontsize="large", ha="left", wrap=True)

        # Save the plot
        # plot_path = f"signal_comparison_window_{i}.png"
        plot_path = f"signal_comparison_window_20.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()        
        
        # plt.figure(figsize=(10, 5))
        # plt.plot(window_data, label="Original Signal", color="blue", alpha=0.7)
        # plt.plot(reconstructed_signal, label="Reconstructed (Top 3 Frequencies)", color="red", linestyle="dashed")
        # plt.legend()
        # plt.title(f"Original vs. Reconstructed Signal for Window {i}")
        # plt.xlabel("Time")
        # # Save plot with a filename based on the window index
        # filename = f"signal_comparison_window_{i}.png"
        # plt.savefig(filename, dpi=300)
        # plt.close()  # Close the figure to prevent memory leaks

fft_features_df = pd.json_normalize(fft_features_list)

df = pd.concat([df.reset_index(drop=True), fft_features_df.reset_index(drop=True)], axis=1)

print(df.tail())