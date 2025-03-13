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


dohlcav_mpnxp_data = pd.read_csv('/lrn/FRC_Bot/FRC-Fourier_Regression_Correction.Bot-API_3.2/predicted_corrected_targets.csv')
# print(dohlcav_mpnxp_data.head())
# sys.exit()

df = pd.DataFrame()
df['delta'] = dohlcav_mpnxp_data['raw predicted'] / dohlcav_mpnxp_data['actual'] 

actual = dohlcav_mpnxp_data['actual'].values.tolist()
raw_predicted_targets = dohlcav_mpnxp_data['raw predicted'].values.tolist()
positions_day_number = [1,2,3,4,5]
endpoint = 'http://192.168.190.18:1221/graphql'

def call_apply_FRC(actual,raw_predicted_targets,positions_day_number,endpoint):
    query = f'''
        query {{
            fitApplyFourierCorrection (
                actual: {actual}
                predicted_targets: {raw_predicted_targets}
                positions_day_number: {positions_day_number}
            ) {{
                success,
                error,
                correction_factors,
                corrected_targets,
            }}
        }}
    '''
    headers = {
        "Accept-Encoding": "gzip, deflate, br",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Connection": "keep-alive",
        "DNT": "1",
    }
    response = requests.post(endpoint, json={"query": query}, headers=headers).json()
    return response

frc_response = call_apply_FRC(actual,raw_predicted_targets,positions_day_number,endpoint)
if not frc_response['data']['fitApplyFourierCorrection']['success']:
    error = frc_response['data']['fitApplyFourierCorrection']['error']
    print('Error:', error)
else:
    # print(frc_response['data']['fitApplyFourierCorrection'].keys())
    # print(frc_response['data']['fitApplyFourierCorrection']['corrected_targets'])
    corrected_targets = frc_response['data']['fitApplyFourierCorrection']['corrected_targets']
    correction_factor = frc_response['data']['fitApplyFourierCorrection']['correction_factors']
    print('corrected_targets', corrected_targets[:15])
    print('correction_factors',correction_factor[:15])