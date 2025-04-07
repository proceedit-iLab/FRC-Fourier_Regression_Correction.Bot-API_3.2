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

dohlcav_mpnxp_data = pd.read_csv('/lrn/FRC_Bot/FRC-Fourier_Regression_Correction.Bot-API_3.2/predicted_corrected_targets.csv')
# print(dohlcav_mpnxp_data.tail())
# sys.exit()

df = pd.DataFrame()
df['delta'] = dohlcav_mpnxp_data['raw predicted'] / dohlcav_mpnxp_data['actual'] 
df['date'] = dohlcav_mpnxp_data['date']

# price_mean = df['delta'].mean() # experiment 1 working with mean

def extract_fft_features(prices, top_n_percent=80):

    n = len(prices) 
    fft_values = fft(prices)
    freqs = fftfreq(n)
    # print('Total frequencies : ',len(fft_values))
    # sys.exit()
    
    amplitudes = np.abs(fft_values) #[:n//2] 
    phases = np.angle(fft_values) #[:n//2] 

    sorted_indices = np.argsort(amplitudes)[::-1]  # sorting in desc order
    top_n = max(1, int(len(sorted_indices) * (top_n_percent / 100)))

    # selecting top frequencies
    top_freqs = freqs[sorted_indices][:top_n]
    top_amplitudes = amplitudes[sorted_indices][:top_n]
    top_phases = phases[sorted_indices][:top_n]
    # print('Number of dominant frequencies : ',len(top_freqs))

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
#     # print("extended_period:",t)

#     reconstructed_signal = np.zeros_like(t, dtype=float)
#     for index in top_indices:
#         freq = index / n
#         amplitude = np.abs(original_fft[index]) / n
#         phase = np.angle(original_fft[index])

#         reconstructed_signal += amplitude * np.cos(2 * np.pi * freq * t + phase) # x(t)=Acos(2πft+ϕ)
#         # sys.exit()

#     print("reconstructed signal : ", reconstructed_signal)
#     print("len reconstructed signal : ",len(reconstructed_signal))
#     return reconstructed_signal[-n_future:] 
#     # return reconstructed_signal[:1]

def reconstruct_signal_from_fft(original_fft, top_indices, n_future=1):
    n = len(original_fft)
    extended_fft = np.zeros(n + n_future, dtype=complex)

    for index in top_indices:
        extended_fft[index] = original_fft[index]

    # extended_fft[:n] = original_fft  # Copy original FFT coefficients

    # Reconstruct the signal using IFFT
    reconstructed_signal = np.fft.ifft(extended_fft).real
    # print("reconstructed signal : ", reconstructed_signal)
    # print("len reconstructed signal : ",len(reconstructed_signal))

    return reconstructed_signal[-n_future:]  # Return extrapolated values

def plot_dominant_frequencies(original_fft, top_indices, n, window_index):
 
    # t = np.arange(n + 1)  
    n = window_size  
    t = np.linspace(0, n, num=n * 10)
    
    plt.figure(figsize=(12, 6))

    for index in top_indices:
        freq = index / n
        amplitude = np.abs(original_fft[index]) / n
        phase = np.angle(original_fft[index])

        signal_component = amplitude * np.cos(2 * np.pi * freq * t + phase)

        plt.plot(t, signal_component, label=f"Freq {freq:.4f} (Index {index})")

    plt.title(f"Top Dominant Frequencies Used for Prediction (Window {window_index})", fontsize=14)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.savefig(f"dominant_frequencies_window_{window_index}.png", dpi=300)
    plt.close()

# - - - - - - x - x - x - x - x - - - - - - - #

window_size = 15  
fft_features_list = [0] * window_size
reconstructed_signals = []
predicted_values = []

'''
for i in range(window_size, len(df)):  
    window_data = df['delta'].iloc[i - window_size:i].values

    # print('window data: ', window_data)
    features, fft_values, top_indices = extract_fft_features(window_data, top_n_percent=80)
    fft_features_list.append(features)  

    next_value = reconstruct_signal_from_fft(fft_values, top_indices, n_future=1)[0]
    predicted_values.append(next_value)
'''
for i in range(window_size, len(df)):
    window_data = df['delta'].iloc[i - window_size:i].values

    input_window = list(window_data[:window_size-5])
    for step in range(6):
        _, fft_values, top_indices = extract_fft_features(np.array(input_window), top_n_percent=100)

        next_value = reconstruct_signal_from_fft(fft_values, top_indices, n_future=1)[0]

        input_window.append(next_value)

    predicted_values.append(input_window[-1])
    features, _, _ = extract_fft_features(np.array(window_data[:window_size-5]), top_n_percent=80)
    fft_features_list.append(features)

    # if i == 15:
    #     sys.exit()

    # if i == 10 : # plot dominant frequencies 
    #     plot_dominant_frequencies(fft_values, top_indices[1:], window_size, i)

    # Plot original vs reconstructed 
    # if i == 10:
    #     trend_slope, trend_intercept, trend_r2, dispersion = sir_parameters(window_data, reconstructed_signal)
        
    #     x_index = np.arange(len(window_data))
    #     # Create the plot
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     sns.lineplot(x=x_index, y=window_data, label="Actual", ax=ax)
    #     sns.lineplot(x=x_index, y=reconstructed_signal, label="Reconstructed", ax=ax)

    #     # Set plot labels and title
    #     ax.set_xlabel('Index', fontsize=12)
    #     ax.set_ylabel('Error', fontsize=12)
    #     ax.set_title(f'Comparison of Original vs Reconstructed', fontsize=14, fontweight='bold')
    #     ax.legend(fontsize=10)

    #     # Add calculated statistics to the plot
    #     text_x = 1.02  # position outside the right edge of the plot
    #     text_y = 0.9   # starting vertical position
    #     text_spacing = 0.07  # space between text lines
    #     stats = [
    #         f"Trend Slope: {trend_slope:.4f}",
    #         f"Trend Intercept: {trend_intercept:.4f}",
    #         f"Trend R²: {trend_r2:.4f}",
    #         f"Standard Deviation: {np.std(window_data):.4f}",
    #         f"Dispersion: {dispersion:.4f}",
    #     ]
    #     for j, stat in enumerate(stats):
    #         plt.figtext(text_x, text_y - j * text_spacing, stat, fontsize="large", ha="left", wrap=True)

        # Save the plot
        # plot_path = f"signal_comparison_window_{i}.png"
        # plot_path = f"signal_comparison_window_20.png"
        # plt.savefig(plot_path, bbox_inches='tight')
        # plt.close()        
        
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
# sys.exit()

# fft_features_df = pd.json_normalize(fft_features_list) # Convert FFT features into a DataFrame

# df = pd.concat([df.reset_index(drop=True), fft_features_df.reset_index(drop=True)], axis=1)

df['predicted_next_value'] = [0] * window_size + predicted_values
# df['predicted_next_value'] = predicted_values + [0] * window_size

print(df.tail())
# df.to_csv('/lrn/fft-analysis/actual_vs_predicted.csv')
# sys.exit()

# Plot the comparison plot between actual and predicted errors
trend_slope, trend_intercept, trend_r2, dispersion = sir_parameters(df['delta'][400:500], df['predicted_next_value'][400:500]) 
x_index = np.arange(len(df['delta'][400:500]))
# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(x=x_index, y=df['delta'][400:500], label="Actual", ax=ax)
sns.lineplot(x=x_index, y=df['predicted_next_value'][400:500], label="Predicted", ax=ax)

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

plot_path = f"signal_real_predicted_comparison.png"
plt.savefig(plot_path, bbox_inches='tight')
plt.close()        