import os

# data access and manipulation modules
import pandas as pd 
import numpy as np
import csv

# plotting libraries
import matplotlib.pyplot as plt
import plotly.graph_objects as go # interactive plots
import matplotlib.dates as mdates
import datetime
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages # saves multiple individual plots in 1 doc in case of not faceted plot
from collections import defaultdict # for plotting layers of each station

def process_data_files(directory):
    data_dict = {}
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        station_name = os.path.splitext(os.path.basename(file_path))[0].replace('_daily_avg', '')

        time = []
        data = []

        with open(file_path, 'r') as datafile:
            plotting = csv.reader(datafile, delimiter='\t')
            next(plotting)  # skips header row
            for ROWS in plotting:
                time.append(datetime.strptime(ROWS[0], '%Y-%m-%d'))
                data.append(float(ROWS[1]))

        data_dict[station_name] = {'time': time, 'data': data}

    return data_dict

def log_transfo_data_dict(original_data_dict):
    log_transformed_dict = {}

    for station_name, data_info in original_data_dict.items():
        log_transformed_data = np.log(data_info['data'])
        log_transformed_dict[station_name] = {'time': data_info['time'], 'data': log_transformed_data}

    return log_transformed_dict

def metrics_to_csv(metrics_dict, name1, name2):

    path = 'C:/Users/violi/Documents/ESDSRS/STD/soil-moisture-ghc/results-test/'
    file_name = f"{name1}_{name2}_four_levels_only_metrics.csv"
    file_path = path+file_name
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        header = ['Station', 'Correlation Coefficient', 'R-squared', 'Slope', 'Intercept']
        writer.writerow(header)


        for station in metrics_dict.keys():
            inner_dict = metrics_dict[station][next(iter(metrics_dict[station]))]
            row = [station, inner_dict['correlation_coefficient'], inner_dict['r_squared'], inner_dict['slope'], inner_dict['intercept']]
            writer.writerow(row)

    print(f'Successfully saved to {file_path}')

def filter_level_data_common_time(dict1, dict2):    # order is important! dct1 is level data ie swc and dict2 is non leveled data ie reco
    dict1_filtered = {}
    dict2_filtered = {}

    for station_name, data1 in dict2.items():
        station_prefix = station_name.split('_')[0]
      #  print(station_name)
        for station_level_name, data2 in dict1.items():
            if station_level_name.startswith(station_prefix):
                level_name = station_level_name.split('_')[2] # the _l1_ etc bit of the name
                # print(level_name)
                #common_time_range = set(data1.get('time', [])).intersection(set(data2.get('time', [])))
                common_time_range = [time for time in data1.get('time', []) if time in data2.get('time', [])]

               # print(common_time_range)
            
                if common_time_range:

                    dict1_key = f"{station_level_name}"
                    dict1_filtered[dict1_key] = {'time': list(common_time_range), 'data': []}

                    dict1_filtered[dict1_key]['data'] = [value for time, value in zip(data2['time'], data2['data']) if time in common_time_range]
                    #print(len(dict1_filtered[dict1_key]['data']))

                    dict2_key = f"{station_name}_{level_name}"  # make new name for reco with level in it
                    dict2_filtered[dict2_key] = {'time': list(common_time_range), 'data': []}

                    dict2_filtered[dict2_key]['data'] = [value for time, value in zip(data1['time'], data1['data']) if time in common_time_range]
#                    print(len(dict2_filtered[dict2_key]['data']))

    return dict1_filtered, dict2_filtered

def calculate_regression_metrics(x, y):
    x = np.array(x)
    y = np.array(y)

    # Find indices where either x or y is NaN
    nan_indices = np.isnan(x) | np.isnan(y)

    # Exclude NaN values from both x and y
    x_valid = x[~nan_indices]
    y_valid = y[~nan_indices]

    if len(x_valid) == 0 or len(y_valid) == 0:
        # If no valid data points are available, return NaN for all metrics
        return np.nan, np.nan, np.nan, np.nan

    correlation_coefficient = np.corrcoef(x_valid, y_valid)[0, 1]

    # Linear regression for slope and intercept
    slope, intercept = np.polyfit(x_valid, y_valid, 1)

    residuals = y_valid - (slope * x_valid + intercept)
    ss_residual = np.sum(residuals**2)
    ss_total = np.sum((y_valid - np.mean(y_valid))**2)
    r_squared = 1 - (ss_residual / ss_total)

    return correlation_coefficient, r_squared, slope, intercept

def calculate_and_store_all_levels(dict1, dict2): 
    all_levels_metrics = {}

    for key1, values1 in dict1.items():
        split_parts = key1.split('_')
        station_name = split_parts[0]
        level_name = split_parts[2]
        matching_keys = [key2 for key2 in dict2.keys() if key2.startswith(station_name)]

        if matching_keys:
            for key2, values2 in dict2.items():
                if key2.startswith(station_name) and 'data' in values2:
                    x = np.array(values1.get('data', []))
                    y = np.array(values2.get('data', []))

                    #print(f'length is {len(x)} for {key2}')

                    if not (np.iterable(x) and np.iterable(y)):
                        print(f"Invalid data format for {key1} or {key2}")
                        continue

                    if len(x) == 0 or len(y) == 0:
                        print(f"Empty data array for {key1} or {key2}")
                        continue

                    min_len = min(len(x), len(y))
                    x = x[:min_len]
                    y = y[:min_len]
                   # print(f'calculating regression metrics at station {key1} {key2}')# for time {x} {y}')

                    correlation_coefficient, r_squared, slope, intercept = calculate_regression_metrics(x, y)

                    level_metrics = {
                        'correlation_coefficient': correlation_coefficient,
                        'r_squared': r_squared,
                        'slope': slope,
                        'intercept': intercept
                    }

                    station_and_level = f"{station_name}_{level_name}"

                    if station_and_level not in all_levels_metrics:
                        all_levels_metrics[station_and_level] = {}

                    all_levels_metrics[station_and_level][station_name] = level_metrics

    return all_levels_metrics

def remove_missing_time_steps(gst_filtered):
    for station_name, levels_data in gst_filtered.items():
        station_prefix = station_name.split('_')[0]

        level_keys = [key for key in gst_filtered if key.startswith(station_prefix)]

        if len(level_keys) < 2:
            continue

        # create a dictionary to store time data for each level
        time_data = {level: gst_filtered[level]['time'] for level in level_keys}

        reference_level = level_keys[0]  # first level as the reference
        for current_level in level_keys[1:]:
            # Update time for the current level
            i = 0  # initialize index
            while i < len(time_data[current_level]):
                timestep = time_data[current_level][i]

                if timestep not in time_data[reference_level]:
                   # print(f'Removing timestep {timestep} from {current_level}')

                    # Remove corresponding 'data' value
                    del gst_filtered[current_level]['data'][i]

                    # Remove the timestep from the list
                    del time_data[current_level][i]
                else:
                    i += 1  # move to the next timestep

            # Update gst_filtered with the modified time data
            gst_filtered[current_level]['time'] = time_data[current_level]

    return gst_filtered

def four_levels_filter(data_dict, data_name):
    desired_levels_stations = {}

    for station_name in set(station.split('_')[0] for station in data_dict.keys()):
        # initialize a dictionary for the station
        station_data = {}

        for station in data_dict:
            if station.startswith(station_name):
                # extract level from the station name
                level = station.split('_')[-1][-2:]

                # get data and time_steps from the gst_filtered dictionary
                data = data_dict[station].get('data', [])
                time_steps = data_dict[station].get('time', [])

                # store data and time_steps in the station_data dictionary
                station_data[level] = {'data': data, 'time': time_steps}

        # check if the station has exactly 4 levels before storing in the dictionary
        if len(station_data) == 4:
            # Find common time steps among all levels
            common_time_steps = set.intersection(*[set(station_data[level]['time']) for level in station_data])

            # Filter data and time_steps to keep only common time steps
            for level in station_data:
                # Filter data and time_steps to keep only common time steps
                station_data[level]['data'] = [value for value, ts in zip(station_data[level]['data'], station_data[level]['time']) if ts in common_time_steps]
                station_data[level]['time'] = list(common_time_steps)

            desired_levels_stations[station_name] = station_data

            
 # Create a dictionary with the desired structure
    four_levels_only_gst_dict = {}
    
    for station_name, levels_data in desired_levels_stations.items():
        for level, data in levels_data.items():
            new_key = f"{station_name}_{data_name}_{level}"
            four_levels_only_gst_dict[new_key] = {'time': data['time'], 'data': data['data']}
    
    return four_levels_only_gst_dict
