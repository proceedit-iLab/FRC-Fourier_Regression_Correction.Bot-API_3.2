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
from scipy.fftpack import fft, ifft

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

features_targets_input_data,error = fetch_features_targets_input_data(case,dataset_start_date,connection_params)
dohlcav_mpnxp_data = features_targets_input_data.copy()
# print(dohlcav_mpnxp_data.tail())
dohlcav_mpnxp_data= dohlcav_mpnxp_data.iloc[::-1].reset_index(drop=True)
# print(dohlcav_mpnxp_data.head())
# sys.exit()

df = pd.DataFrame()

# Compute daily price change (Delta)
df['ACPCP_adjusted_close_price_current_period'] = dohlcav_mpnxp_data['ACPCP_adjusted_close_price_current_period']
# df['delta'] = dohlcav_mpnxp_data['ACPCP_adjusted_close_price_current_period'].diff().fillna(0)
df['delta'] = dohlcav_mpnxp_data['ACPCP_adjusted_close_price_current_period']

price_mean = df['ACPCP_adjusted_close_price_current_period'].mean() # exp.1 working with mean

# fft_result = np.fft.fft(df['delta'])
fft_result = np.fft.fft(df['delta'] - price_mean)
frequencies = np.fft.fftfreq(len(fft_result), d=1) 
magnitude = np.abs(fft_result)
periods = 1 / frequencies

# Simple visualization of price
plt.figure(figsize=(10, 8))

# Time series plot
plt.subplot(2, 1, 1)
plt.plot(df.index, df['delta'])
plt.title('TSLA Time Series')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')

# Frequency domain representation
plt.subplot(2, 1, 2)
plt.plot(periods, magnitude)
plt.title('FFT of TSLA')
plt.xlabel('Period (Days)')
plt.ylabel('Magnitude')
plt.xlim(0, max(periods[1:]))  # Limiting x-axis to view significant periods
plt.ylim(0, max(magnitude[1:]) * 1.1)  # Ignore the zero frequency component

plt.tight_layout()
plt.savefig('tsla_fft_analysis.png')  # Save plot
plt.close()

# recovered = np.fft.ifft(fft_result)
recovered = np.fft.ifft(fft_result) + price_mean


plt.figure(figsize=(14, 6))
plt.plot(df.index, df['delta'], label='Original')
plt.plot(df.index, recovered, label='Recovered', linestyle='dashed')
plt.title('TSLA Time Series: Full FFT Inverse')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.savefig('tsla_fft_reconstruction.png')  # Save plot
plt.close()

# Extracting dominant frequencies
# dominant_periods = pd.Series(periods, index=magnitude).nlargest(100)

# Recovering with dominant frequencies
top_fft_result = fft_result.copy()
threshold = np.percentile(magnitude, 99.9)  # Keep top (100-n)% frequencies
top_fft_result[magnitude < threshold] = 0  # Zero-out weaker frequencies

top_recovered = np.fft.ifft(top_fft_result) + price_mean  # Add back mean

# Recovering the original time series with just the top 25 dominant periods

# top_periods = dominant_periods.index 
# top_fft_result = fft_result.copy()
# top_fft_result[np.abs(frequencies) > 1 / top_periods.min()] = 0 #original, makes frequencies 0

# top_recovered = np.fft.ifft(top_fft_result)

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['delta'], label='Original')
plt.plot(df.index, top_recovered, label='Reconstructed (Filtered IFFT)', linestyle='dashed')
plt.title('TSLA Time Series: FFT Inverse with Dominant Frequencies')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.savefig('tsla_fft_filtered_reconstruction.png')  # Save plot
plt.close()

