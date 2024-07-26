# data access and manipulation modules
import pandas as pd # mostly used to convert data into a workable format  i.e. a dataframe
import numpy as np # maths module
import math
import csv # to read csv files
import os

# plotting libraries
import matplotlib.pyplot as plt
import plotly.graph_objects as go # interactive plots
from plotly.subplots import make_subplots # for dropdown menu options in teractive plots!
import matplotlib.dates as mdates # for plotting dates nicely
import tabulate # to make markdown formatted table
import datetime
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages # saves multiple individual plots in 1 doc in case of not faceted plot
from collections import defaultdict # for plotting layers of each station

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def process_data_files(directory):
    '''
    Description:
        This function stitches together in individual stations' data and returns a dictionary with the station names as the keys, and the data (time, SWC/RECO) as the value. It is the first necessary step to getthe .txt format of the data into a more flexibile one to work with in the analysis. 

    Parameters:
        directory - A directory from which the files will be read in

    Return:
        data_dict - A dictionary of the file contents
    '''


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
    '''
    Description:
        This function takes the data dictionary created in the process_data_files function, and returns a dictionary, with the variable of interest being log transformed. It is used later for metric calculations, since it is sometimes interesting to look at the log transform of the variable..

    Parameters:
        data dictionary - a dictionary containing the data. Must be created by the proccess_data_files function, or have the same strucutre: data_dict = {station_name_level: {time: [datetime_list], data: [float_list]}}

    Return:
        data_dict - A dictionary of the file contents the same as the input parameter, just with the data variable having a log transformaiton applied
    '''
    log_transformed_dict = {}

    for station_name, data_info in original_data_dict.items():
        log_transformed_data = np.log(data_info['data'])
        log_transformed_dict[station_name] = {'time': data_info['time'], 'data': log_transformed_data}

    return log_transformed_dict


def remove_missing_time_steps(dict1):
    '''
Description:
    This function processes a dictionary of FLUXNET data to remove time steps that are missing in some levels but present in others. It ensures time consistency across different measurement levels for each station. The function operates on the data dictionary in-place, modifying it to maintain only time steps that are present in all levels of a given station.

Parameters:
    dict1 - A dictionary containing FLUXNET data. The structure is expected to be:
            {station_name_level: {'time': [datetime_list], 'data': [float_list]}}
            where station_name_level is a string combining station name and measurement level.

Return:
    dict1 - The same dictionary as input, but with missing time steps removed to ensure consistency across levels for each station. The structure remains the same, but some entries in 'time' and 'data' lists may have been removed.

Key Operations:
    1. Groups data by station prefix.
    2. Uses the first level of each station as a reference for time steps.
    3. Compares other levels to the reference, removing time steps (and corresponding data) that don't exist in the reference level.
    4. Updates the dictionary in-place, ensuring all levels for a station have consistent time steps.

'''
    for station_name, levels_data in dict1.items():
        station_prefix = station_name.split('_')[0]

        level_keys = [key for key in dict1 if key.startswith(station_prefix)]

        if len(level_keys) < 2:
            continue

        # create a dictionary to store time data for each level
        time_data = {level: dict1[level]['time'] for level in level_keys}

        reference_level = level_keys[0]  # first level as the reference
        for current_level in level_keys[1:]:
            # Update time for the current level
            i = 0  # initialize index
            while i < len(time_data[current_level]):
                timestep = time_data[current_level][i]

                if timestep not in time_data[reference_level]:

                    # Remove corresponding 'data' value
                    del dict1[current_level]['data'][i]

                    # Remove the timestep from the list
                    del time_data[current_level][i]
                else:
                    i += 1  # move to the next timestep

            # Update dictionary with the modified time data
            dict1[current_level]['time'] = time_data[current_level]

    return dict1


def four_levels_filter(data_dict, data_name):
    '''
Description:
    This function processes a dictionary of FLUXNET data to filter and restructure it, keeping only stations with exactly four depth measurement levels. It ensures that all levels within a station have consistent time steps. The function creates a new dictionary with a standardized structure for further analysis.

Parameters:
    data_dict - A dictionary containing FLUXNET data. The structure is expected to be:
                {station_name_level: {'time': [datetime_list], 'data': [float_list]}}
    data_name - A string representing the name of the data variable being processed.

Return:
    four_levels_only_gst_dict - A new dictionary containing only stations with four levels, 
                                restructured for consistency. The new structure is:
                                {station_name_data_name_level: {'time': [datetime_list], 'data': [float_list]}}. 'gst' in the return statement is an artifact from testing the function that I didn' remove. Please ignore the name there. 

Key Operations:
    1. Groups data by station name.
    2. Filters to keep only stations with exactly four measurement levels.
    3. Finds common time steps across all levels for each station.
    4. Filters data to include only these common time steps, ensuring temporal consistency.
    5. Restructures the data into a new dictionary with a standardized format.

'''

    desired_levels_stations = {}

    for station_name in set(station.split('_')[0] for station in data_dict.keys()):
        # initialize a dictionary for the station
        station_data = {}

        for station in data_dict:
            if station.startswith(station_name):
                # extract level from the station name
                level = station.split('_')[-1][-2:]

                # get data and time_steps from the dictionary
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


def filter_level_data_common_time(dict1, dict2):    # order is important! dct1 is level data ie swc and dict2 is non leveled data ie reco
    '''
Description:
    This function aligns two FLUXNET datasets: one resulting from the four_levels_filter function (dict1) 
    and another dataset (dict2) that may not have level-specific data. It ensures that both datasets 
    have matching time steps for each station and level, allowing for comparative analysis.

Parameters:
    dict1 - A dictionary containing level-specific FLUXNET data, output from the four_levels_filter function.
            Structure: {station_name_data_name_level: {'time': [datetime_list], 'data': [float_list]}}
    dict2 - A dictionary containing non-level-specific FLUXNET data.
            Structure: {station_name: {'time': [datetime_list], 'data': [float_list]}}

Return:
    dict1_filtered - technically no change with inpout dict1 - redefined here for naming consistency.
    dict2_filtered - A new dictionary derived from dict2, restructured to match dict1's format and 
                     filtered to include only common time steps with dict1.
                     New structure: {station_name_level: {'time': [datetime_list], 'data': [float_list]}}

Key Operations:
    1. Iterates through stations in dict2 and matches them with corresponding levels in dict1.
    2. Identifies common time steps between matched datasets.
    3. Filters both datasets to include only these common time steps.
    4. Restructures dict2 to match the level-specific format of dict1.

Note:
    This function is crucial for preparing FLUXNET data for comparative analysis between different 
    types of measurements, especially when one dataset has been pre-processed for four-level consistency. 
    It ensures temporal alignment between level-specific and level-free data, allowing for 
    accurate comparisons across different variables and measurement levels.
'''
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
    '''
    Description:
        This function calculates the regression metrics between 2 given variables, x and y. It is used later in a function that stored these values in a meaningful way.

    Parameters:
        x - the data part of the data dictionaries - dict1[key1]['data']
        y - same as x, but a different data

    Return:
        metric1, metric2, metric3, metric 4 - 4 different metrics (correlation coefficient, r squared, slope and intercept)
    '''
    x = np.array(x)
    y = np.array(y)

    # find indices where either x or y is NaN
    nan_indices = np.isnan(x) | np.isnan(y)

    # exclude NaN values from both x and y
    x_valid = x[~nan_indices]
    y_valid = y[~nan_indices]

    if len(x_valid) == 0 or len(y_valid) == 0:
        # if no valid data points are available, return NaN for all metrics
        return np.nan, np.nan, np.nan, np.nan

    correlation_coefficient = np.corrcoef(x_valid, y_valid)[0, 1]

    # linear regression for slope and intercept
    slope, intercept = np.polyfit(x_valid, y_valid, 1)

    residuals = y_valid - (slope * x_valid + intercept)
    ss_residual = np.sum(residuals**2)
    ss_total = np.sum((y_valid - np.mean(y_valid))**2)
    r_squared = 1 - (ss_residual / ss_total)

    return correlation_coefficient, r_squared, slope, intercept

def calculate_and_store_all_levels(dict1, dict2): 
    '''
    Description:
        This function calculates the regression metrics between 2 given variables, x and y. It is used later in a function that stored these values in a meaningful way.

    Parameters:
        dict1 - the data dictionaries as defined by 
        dict 2 -

    Return:
        metric1, metric2, metric3, metric 4 - 4 different metrics (correlation coefficient, r squared, slope and intercept)
    '''
    all_levels_metrics = {} # initialise emoty dictionary

    for key1, values1 in dict1.items():
      # the next 3 lines are to get the station name prefix out of the first dictionary
        split_parts = key1.split('_')
        station_name = split_parts[0]
        level_name = split_parts[2]
        matching_keys = [key2 for key2 in dict2.keys() if key2.startswith(station_name)]

        if matching_keys:
            for key2, values2 in dict2.items():
                if key2.startswith(station_name) and 'data' in values2:
                    x = np.array(values1.get('data', []))
                    y = np.array(values2.get('data', []))

                    if not (np.iterable(x) and np.iterable(y)):
                        print(f"Invalid data format for {key1} or {key2}")
                        continue

                    if len(x) == 0 or len(y) == 0:
                        print(f"Empty data array for {key1} or {key2}")
                        continue

                    min_len = min(len(x), len(y))
                    x = x[:min_len]
                    y = y[:min_len]

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


def metrics_to_csv(metrics_dict, name1, name2):
    '''
Description:
    This function takes a dictionary of calculated metrics and writes them to a CSV file. It's designed 
    to export statistical metrics (such as correlation coefficient, R-squared, slope, and intercept) 
    for different stations, likely derived from comparing two FLUXNET variables.

Parameters:
    metrics_dict - A nested dictionary containing calculated metrics for each station.
                   Structure: {station: {metric_set: {metric_name: value}}}
    name1 - A string, likely representing the name of the first variable being compared.
    name2 - A string, likely representing the name of the second variable being compared.

Return:
    None - The function doesn't return a value, but it creates a CSV file as a side effect.

Key Operations:
    1. Constructs a file name and path based on the input parameters.
    2. Opens a new CSV file for writing.
    3. Writes a header row with column names.
    4. Iterates through the metrics dictionary, extracting values for each station.
    5. Writes a row for each station with its corresponding metric values.
    6. Prints a confirmation message with the file path upon successful completion.

'''


    path = '/path/to/where/you/want/to/save/the/csv/'
    file_name = f"{name1}_{name2}_something_else.csv"
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

def create_world_correlation_map(df, metrics, levels, data_type): # for now, ave to manually type in the variable for the title. want to imrove this to be dynamic later.
    '''
Description:
    This function creates an interactive world map using Plotly to visualize correlation metrics between 2 variables  - i.e. soil water content (SWC) and ecosystem respiration (RECO) across different measurement levels. The map includes a dropdown menu to select different metrics and levels, updating the map dynamically based on the selection.

Parameters:
    df - A Pandas DataFrame containing the data to be plotted. The DataFrame should include columns for longitude ('lon'), latitude ('lat'), station names ('station'), levels ('level'), and the metrics to be visualized.
    metrics - A list of strings representing the names of the metrics to be visualized (e.g., ['correlation_coefficient', 'r_squared']).
    levels - A list of strings or integers representing the measurement levels (e.g., ['L1', 'L2', 'L3', 'L4']).
    data_type - A string representing the type of data being compared with SWC (e.g., 'RECO').

Return:
    fig - A Plotly Figure object containing the interactive world map with dropdown menu for selecting metrics and levels.

Key Operations:
    1. Creates a base layer with all data points in light grey.
    2. Adds traces for each combination of metric and level, initially hidden.
    3. Configures a dropdown menu to toggle the visibility of traces based on the selected metric and level.
    4. Sets up the map layout, including geographic settings, size, margins, and annotations.
    5. Adds initial title and dropdown label annotations.

'''
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scattergeo'}]])

    # Create a base layer with all data points
    fig.add_trace(go.Scattergeo(
        lon = df['lon'],
        lat = df['lat'],
        mode = 'markers',
        marker = dict(size = 8, color = 'lightgrey'),
        showlegend = False
    ))

    # Create traces for each combination of metric and level
    for metric in metrics:
        for level in levels:
            level_data = df[df['level'] == level]
            fig.add_trace(go.Scattergeo(
                lon = level_data['lon'],
                lat = level_data['lat'],
                text = level_data['station'] + '<br>' + metric + ': ' + level_data[metric].round(3).astype(str),
                mode = 'markers',
                marker = dict(
                    size = 8,
                    color = level_data[metric],
                    colorscale = 'Viridis',
                    showscale = True,
                  #  colorbar_title = metric
                ),
                name = f'{metric} - Level {level}',
                visible = False  # Start with all traces hidden
            ))

    # Make the first trace visible
    fig.data[1].visible = True

    # Create dropdown menu
    dropdown_buttons = []
    for i, metric in enumerate(metrics):
        for j, level in enumerate(levels):
            visible = [False] * len(fig.data)
            visible[0] = True  # Base layer always visible
            visible[1 + i*len(levels) + j] = True  # Make the selected trace visible
            dropdown_buttons.append(
                dict(
                    label = f'{metric.capitalize()} - Level {level}',
                    method = 'update',
                    args = [{'visible': visible},
                            {'annotations[1].text': f'{metric.capitalize()} between SWC and {data_type} for Level {level}, all stations'}]
                )
            )

    # Set the initial title
    initial_title = f'{metrics[0].capitalize()} between SWC and {data_type} for Level {levels[0]}, all stations'

    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=dropdown_buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1,
            xanchor="right",
            y=1.15,  # Position the dropdown menu
            yanchor="top"
        )],
        geo=dict(
            showland=True,
            showcountries=True,
            showocean=True,
            countrywidth=0.5,
            landcolor='rgb(243, 243, 243)',
            oceancolor='rgb(230, 230, 250)',
            projection_type='natural earth',
            showcoastlines=True,
            coastlinecolor="RebeccaPurple",
            showframe=False,
            lonaxis=dict(showgrid=False),
            lataxis=dict(showgrid=False)
        ),
        autosize=False,
        width=900,
        height=600,  # Increased height to accommodate title and dropdown
        margin=dict(l=0, r=0, t=150, b=0)  # Increased top margin for title and dropdown
    )
    
    # Add annotations for dropdown label and title
    fig.add_annotation(
        text="Select Metric and Level:",
        xref="paper", yref="paper",
        x=0.3, y=1.12,  # Position for dropdown label
        showarrow=False,
        font=dict(size=14)
    )
    
    fig.add_annotation(
        text=initial_title,
        xref="paper", yref="paper",
        x=0.5, y=1.06,  # Position for title
        showarrow=False,
        font=dict(size=16, color="black"),
        align="center"
    )
    
    return fig

def create_timeseries_plot(data, data_type):
    '''
Description:
    This function creates an interactive time series plot using Plotly to visualize FLUXNET data 
    all stations in the input data. It includes a dropdown menu to select different stations, dynamically 
    updating the plot based on the selection.

Parameters:
    data - A nested dictionary containing the time series data. 
           For SWC: {station_datatype_level: {'time': [datetime_list], 'data': [float_list]}}
           For other types: {station: {'time': [datetime_list], 'data': [float_list]}}
    data_type - A string indicating the type of data being plotted ('swc' for soil water content, 
                or another identifier for different data types). Needed for titling the plots (removes a lot of 
                complexity to just write the name rather than extract it from dictionary)

Return:
    fig - A Plotly Figure object containing the interactive time series plot with a dropdown menu 
          for selecting different stations.

Key Operations:
    1. Extracts unique station names from the data keys.
    2. Creates traces for each station and level (for SWC) or just each station (for other data types).
    3. Configures a dropdown menu to toggle the visibility of traces based on the selected station.
    4. Sets up the plot layout, including title, axis labels, and dropdown menu positioning.
    5. Makes the first station's data visible by default.


Note:
    The function handles two different data structures based on the 'data_type' parameter, making it 
    versatile for different types of FLUXNET data visualization.
'''
    # Extract unique stations from the keys
    if data_type == 'swc':
        stations = list(set([key.split('_')[0] for key in data.keys()]))
    else:
        stations = list(data.keys())
    
    # Create the figure
    fig = go.Figure()

    # Create traces for each station (initially hidden)
    for station in stations:
        if data_type == 'swc':
            for level in ['l1', 'l2', 'l3', 'l4']:
                key = f"{station}_{data_type}_{level}"
                if key in data:
                    fig.add_trace(go.Scatter(
                        x=data[key]['time'],
                        y=data[key]['data'],
                        mode='lines',
                        name=f'{level}',
                        visible=False
                    ))
        else:
            fig.add_trace(go.Scatter(
                x=data[station]['time'],
                y=data[station]['data'],
                mode='lines',
                name=station,
                visible=False
            ))

    # Create and add dropdown menu
    dropdown_buttons = []
    for i, station in enumerate(stations):
        visible = [False] * len(fig.data)
        if data_type == 'swc':
            start_idx = i * 4
            for j in range(4):  # 4 levels per station
                if start_idx + j < len(fig.data):
                    visible[start_idx + j] = True
        else:
            visible[i] = True
        
        dropdown_buttons.append(dict(
            label=station,
            method='update',
            args=[{'visible': visible},
                  {'title': f'{data_type.upper()} Time Series for Station {station}'}]
        ))

    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=dropdown_buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )],
        height=600,
        title_text=f"Select a Station to view {data_type.upper()} Time Series",
        title_x=0.5,
        xaxis_title="Time",
        yaxis_title=data_type.upper()
    )

    # Make the first station visible by default
    if data_type == 'swc':
        for i in range(min(4, len(fig.data))):
            fig.data[i].visible = True
    else:
        fig.data[0].visible = True

    return fig


def create_seasonal_plot_multi_level(data, data_type):

    '''
Description:
    This function creates an interactive seasonal plot using Plotly to visualize FLUXNET data 
    across multiple levels for different stations. It generates 
    a subplot for each soil level, showing monthly averages and standard deviations.

Parameters:
    data - A nested dictionary containing the time series data.
           Structure: {station_datatype_level: {'time': [datetime_list], 'data': [float_list]}}
    data_type - A string indicating the type of data being plotted (e.g., 'swc' for soil water content).

Return:
    fig - A Plotly Figure object containing the interactive seasonal plot with subplots for each level 
          and a dropdown menu for selecting different stations.

Key Operations:
    1. Creates a subplot for each of the four levels (L1, L2, L3, L4).
    2. For each station and level, calculates monthly statistics (mean and standard deviation).
    3. Plots mean values as lines with markers and standard deviations as error bars.
    4. Configures a dropdown menu to toggle visibility of traces for different stations.
    5. Sets up the plot layout, including titles, axis labels, and legend positioning.


Note:
    The function assumes four levels of data for each station. It's designed to handle multiple 
    stations, allowing for comparative analysis of seasonal patterns across different locations along with the level information.
'''
    stations = list(set([key.split('_')[0] for key in data.keys()]))
    
    fig = make_subplots(rows=4, cols=1, subplot_titles=("Level 1", "Level 2", "Level 3", "Level 4"),
                        shared_xaxes=True, vertical_spacing=0.05)
    levels = ['l1', 'l2', 'l3', 'l4']

    for station in stations:
        for i, level in enumerate(levels):
            key = f"{station}_{data_type}_{level}"
            if key in data:
                df = pd.DataFrame({
                    'time': pd.to_datetime(data[key]['time']),
                    'value': data[key]['data']
                })
                df['month'] = df['time'].dt.month
                df = df.dropna(subset=['value'])

                monthly_stats = df.groupby('month')['value'].agg(['mean', 'std']).reset_index()

                # Plot mean line
                fig.add_trace(go.Scatter(
                    x=monthly_stats['month'],
                    y=monthly_stats['mean'],
                    mode='lines+markers',
                    name=f'{station} - Mean',
                    line=dict(color='blue', width=2),
                    visible=False
                ), row=i+1, col=1)

                # Add error bars
                fig.add_trace(go.Scatter(
                    x=monthly_stats['month'],
                    y=monthly_stats['mean'],
                    error_y=dict(
                        type='data',
                        array=monthly_stats['std'],
                        visible=True
                    ),
                    mode='markers',
                    marker=dict(color='red', size=8),
                    name=f'{station} - Std Dev',
                    showlegend=False,
                    visible=False
                ), row=i+1, col=1)

    # Create dropdown menu
    dropdown_buttons = []
    for i, station in enumerate(stations):
        visible = [False] * len(fig.data)
        for j in range(8):  # 4 levels * 2 traces per level
            if i*8 + j < len(fig.data):
                visible[i*8 + j] = True
        
        dropdown_buttons.append(dict(
            label=station,
            method='update',
            args=[{'visible': visible},
                  {'title': f'Seasonal Variation of {data_type.upper()} for Station {station}'}]
        ))

    fig.update_layout(
        height=1000,
        title_text=f"Seasonal Variation of {data_type.upper()}",
        showlegend=True,
        updatemenus=[dict(
            active=0,
            buttons=dropdown_buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(
        title_text="Month", 
        row=4, col=1, 
        tickmode='array', 
        tickvals=list(range(1, 13)),
        ticktext=month_names
    )
    for i in range(1, 5):
        fig.update_yaxes(title_text=f"SWC - Level {i}", row=i, col=1)

    # Make the first station visible by default
    for i in range(min(8, len(fig.data))):
        fig.data[i].visible = True

    return fig

def create_seasonal_plot_single_level(data, data_type):
    '''
    Description:
        This function creates an interactive seasonal plot using Plotly to visualize FLUXNET data 
        (typically soil water content or another data type) for multiple stations at a single measurement level. 
        It generates a plot showing monthly averages and standard deviations for each station.

    Parameters:
        data - A nested dictionary containing the time series data.
            Structure: {station_datatype_level: {'time': [datetime_list], 'data': [float_list]}}
        data_type - A string indicating the type of data being plotted (e.g., 'swc' for soil water content).

    Return:
        fig - A Plotly Figure object containing the interactive seasonal plot with a dropdown menu 
            for selecting different stations.

    Key Operations:
    
        1. For each station and level, calculates monthly statistics (mean and standard deviation).
        2. Plots mean values as lines with markers and standard deviations as error bars.
        3. Configures a dropdown menu to toggle visibility of traces for different stations.
        4. Sets up the plot layout, including titles, axis labels, and legend positioning.


    Usage:
        This function is useful for visualizing seasonal patterns of level-free data across multiple FLUXNET stations. 


    '''

    # Extract unique station prefixes
    stations = list(set([key.split('_')[0] for key in data.keys()]))
    
    fig = go.Figure()

    for station in stations:
        # Find the first key that matches this station prefix
        station_key = next(key for key in data.keys() if key.startswith(station))
        
        df = pd.DataFrame({
            'time': pd.to_datetime(data[station_key]['time']),
            'value': data[station_key]['data']
        })
        df['month'] = df['time'].dt.month
        df = df.dropna(subset=['value'])

        monthly_stats = df.groupby('month')['value'].agg(['mean', 'std']).reset_index()

        # Plot mean line
        fig.add_trace(go.Scatter(
            x=monthly_stats['month'],
            y=monthly_stats['mean'],
            mode='lines+markers',
            name=f'{station} - Mean',
            line=dict(color='blue', width=2),
            visible=False
        ))

        # Add error bars
        fig.add_trace(go.Scatter(
            x=monthly_stats['month'],
            y=monthly_stats['mean'],
            error_y=dict(
                type='data',
                array=monthly_stats['std'],
                visible=True
            ),
            mode='markers',
            marker=dict(color='red', size=8),
            name=f'{station} - Std Dev',
            visible=False
        ))

    # Create dropdown menu
    dropdown_buttons = []
    for i, station in enumerate(stations):
        visible = [False] * len(fig.data)
        visible[i*2] = True
        visible[i*2 + 1] = True
        
        dropdown_buttons.append(dict(
            label=station,
            method='update',
            args=[{'visible': visible},
                  {'title': f'Seasonal Variation of {data_type.upper()} for Station {station}'}]
        ))

    fig.update_layout(
        height=600,
        title_text=f"Seasonal Variation of {data_type.upper()}",
        showlegend=True,
        updatemenus=[dict(
            active=0,
            buttons=dropdown_buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(
        title_text="Month", 
        tickmode='array', 
        tickvals=list(range(1, 13)),
        ticktext=month_names
    )
    fig.update_yaxes(title_text=f"{data_type.upper()}")

    # Make the first station visible by default
    if len(fig.data) > 0:
        fig.data[0].visible = True
        fig.data[1].visible = True

    return fig
