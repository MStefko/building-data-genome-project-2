from functools import lru_cache

import os
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import multiprocessing

@lru_cache(maxsize=64)
def import_weather_csv(siteid, columns=None) -> pd.DataFrame:
    """
    Import weather data from a csv file and return a pandas dataframe.
    
    The csv file should be in the following format:
    timestamp,site_id,airTemperature,cloudCoverage,dewTemperature,precipDepth1HR,precipDepth6HR,seaLvlPressure,windDirection,windSpeed
    
    The time step of the csv file is 1 hour.
    
    Parameters
    ----------
    file_path : str
        The path to the csv file containing the weather data.
    columns : list, optional
        A list of columns to import from the csv file. The default is None.
        
    Returns
    -------
    pandas.DataFrame
        A pandas dataframe containing the weather data.
    """
    file_path = os.path.join(os.path.dirname(__file__),'..','data/weather/weather.csv')
    result = pd.read_csv(file_path)

    result = result[result['site_id'] == siteid].drop(columns=['site_id'])
    result['timestamp'] = pd.to_datetime(result['timestamp'])
    result.set_index('timestamp', inplace=True)
    if columns is not None:
        result = result[list(columns)]
    return result

@lru_cache(maxsize=64)
def import_weather_timeseries(site: str, target_timestep: str, columns: Optional[List] = None) -> pd.DataFrame:
    """
    Import time series data from a csv file and return a pandas dataframe.
    
    The csv file should be in the following format:
    timestamp,site_id,value
    
    The time step of the csv file is 1 hour.
    
    Parameters
    ----------
    file_path : str
        The path to the csv file containing the time series data.
    columns : list, optional
        A list of columns to import from the csv file. The default is None.
        
    Returns
    -------
    pandas.DataFrame
        A pandas dataframe containing the time series data.
    """
    
    df = import_weather_csv(site, columns=columns)
    
    print("Imported dataframe info:")
    df.info()
    
    print("Resampling to 1 hour target_timestep and filling in missing values by linear interpolation...")
    # resample to 1 hour target_timestep and fill in missing values by linear interpolation
    df = df.resample(target_timestep).asfreq().interpolate(method='linear')
    df.info()
    
    return df

def get_subseries(df: pd.DataFrame, start_time: datetime, length: timedelta) -> pd.DataFrame:
    end_time = start_time + length
    subseries = df.loc[start_time:end_time]
    return subseries

def get_all_disjoint_subseries(df: pd.DataFrame, length: timedelta) -> List[pd.DataFrame]:
    start_time = df.index.min()
    end_time = df.index.max()
    subseries = []
    while start_time < end_time:
        subseries.append(get_subseries(df, start_time, length))
        start_time += length
    return subseries

def load_all_disjoint_weather_subseries_for_site(site: str, target_timestep: str, subseries_length: timedelta, columns: Optional[List] = None) -> List[pd.DataFrame]:
    df = import_weather_timeseries(site, target_timestep, columns=columns)
    return get_all_disjoint_subseries(df, subseries_length)



@lru_cache(maxsize=64)
def load_meter_data_for_building(meter_type: str, building_id: str) -> pd.DataFrame:
    """
    Load meter data for a specific building and meter type.
    
    Parameters
    ----------
    meter_type : str
        The type of meter data to load.
    building_id : str
        The id of the building to load meter data for.
    target_timestep : str
        The target timestep to resample the meter data to.
        
    Returns
    -------
    pandas.DataFrame
        A pandas dataframe containing the meter data for the building.
    """
    meter_data_path = os.path.join(os.path.dirname(__file__),'..',f'data/meters/cleaned/{meter_type}_cleaned.csv')
    df = pd.read_csv(meter_data_path, parse_dates=['timestamp'], index_col='timestamp', usecols=['timestamp', building_id])
    return df

def load_all_meter_data_for_building_for_particular_period(building_id: str, start_time: datetime, duration: timedelta, target_timestep: str, missingval_tolerance: float, columns_force=None) -> pd.DataFrame:
    df = load_all_meter_series_for_building(building_id, target_timestep, missingval_tolerance, columns_force=columns_force)
    return get_subseries(df, start_time, duration)

@lru_cache(maxsize=64)
def load_all_meter_series_for_building(building_id: str, target_timestep: str, missingval_tolerance: float, columns_force=None) -> pd.DataFrame:
    df = load_all_available_meter_types(building_id, missingval_tolerance)
    df.info()
    if columns_force is not None:
        # insert columns that are in columns_force but not in df
        for column in columns_force:
            if column not in df.columns:
                print(f"Building {building_id} has missing data for column {column}, inserting column with zeros...")
                df[column] = 0
        # drop columns that are not in columns_force
        df = df[list(columns_force)]

    
    # resample to target_timestep and fill in missing values by linear interpolation
    #df = df.resample(target_timestep).asfreq().interpolate(method='linear')
    df = df.interpolate(method='linear')
    
    return df

def load_all_disjoint_meter_subseries_for_building(building_id: str, target_timestep: str, subseries_length: timedelta, missingval_tolerance: float, columns_force=None) -> List[pd.DataFrame]:
    df = load_all_meter_series_for_building(building_id, target_timestep, missingval_tolerance, columns_force=columns_force)
    return get_all_disjoint_subseries(df, subseries_length)

def load_all_available_meter_types(building_id: str, missingval_tolerance: float) -> pd.DataFrame:
    all_meter_types = ["electricity", "chilledwater", "hotwater", "steam", "gas", "water", "irrigation", "solar"]
    
    # This section is just to get a good index for the dataframe
    for meter_type in all_meter_types:
        try:
            df = load_meter_data_for_building(meter_type, building_id)
            break
        except ValueError:
            continue
    
        
    # Add all available meter types to the dataframe
    for meter_type in all_meter_types:
        try:
            pdf = load_meter_data_for_building(meter_type, building_id)
        except ValueError:
            print(f"Meter type {meter_type} for building {building_id} not found, skipping...")
            continue
        missingval_rate = pdf.isnull().sum()[building_id] / len(df)
        if missingval_rate > missingval_tolerance:
            print(f"Meter type {meter_type} for building {building_id} has too many missing values ({missingval_rate:.1%}), skipping...")
            continue
        df[meter_type] = pdf[building_id]
    df.drop(columns=[building_id], inplace=True)
    #print(f"Loaded {len(df.columns)} meter types for building {building_id}")
    #df.info()
    return df

def get_weather_data_from_time(site: str, time: datetime, desired_duration: timedelta, target_timestep: str, target_columns: List[str]) -> pd.DataFrame:
    weather_df = import_weather_timeseries(site, target_timestep, columns=target_columns)
    # crop weather_df based on time and desired_duration
    start_time = time
    end_time = time + desired_duration
    weather_df = weather_df.loc[start_time:end_time]
    return weather_df
    
def transform_to_input_batch(weather_df: pd.DataFrame, meter_df: pd.DataFrame, future_weather_df: pd.DataFrame) -> pd.DataFrame:
    # join weather_df, meter_df
    merged_df = weather_df.join(meter_df)
    
    # append weather data from future_weather_df, filling in missing columns (i.e. meter values which are unknown now) with zeros
    merged_df = pd.concat((merged_df,future_weather_df)).fillna(0)
    
    return merged_df

def generate_batches(building_id: str, sampling_rate: str, subseries_length: timedelta, missingval_tolerance: float, prediction_duration: timedelta, weather_columns: Tuple[str], meter_columns: Tuple[str], prediction_variable: str) -> List[Tuple[pd.DataFrame, pd.Series]]:
    site = building_id.split('_')[0]
    weather_dfs = load_all_disjoint_weather_subseries_for_site(site, sampling_rate, subseries_length, columns=weather_columns)
    print(f"Loaded {len(weather_dfs)} disjoint subseries for site {site} with length {subseries_length}")
    
    meter_dfs = load_all_disjoint_meter_subseries_for_building(building_id, sampling_rate, subseries_length, missingval_tolerance, columns_force=meter_columns)
    print(f"Loaded {len(meter_dfs)} disjoint subseries for building {building_id} with length {subseries_length}")
    
    batches = []
    for weather_df, meter_df in zip(weather_dfs, meter_dfs):
        prediction_start = meter_df.index.max()
        future_weather_df = get_weather_data_from_time(site, prediction_start, prediction_duration, sampling_rate, weather_columns)
        future_sensor_df = load_all_meter_data_for_building_for_particular_period(building_id, prediction_start, prediction_duration, sampling_rate, missingval_tolerance, columns_force=meter_columns)
    
        input_batch = transform_to_input_batch(weather_df, meter_df, future_weather_df)
        output_batch = future_sensor_df[prediction_variable]
        batches.append((input_batch, output_batch))
        
    return batches

def generate_batches_building(building, sampling_rate, subseries_length, missingval_tolerance, prediction_duration, weather_columns, meter_columns, prediction_variable):
    print(f"Building: {building}")
    batches = generate_batches(building, sampling_rate, subseries_length, missingval_tolerance, prediction_duration, weather_columns, meter_columns, prediction_variable)
    print(f"Generated {len(batches)} batches")
    return building, batches

def generate_batches_single_argument(arg):
    return generate_batches_building(*arg)

def generate_building_dataframe(building_id, sampling_rate, missingval_tolerance, weather_columns, meter_columns):
    site = building_id.split('_')[0]
    weather_df = import_weather_timeseries(site, sampling_rate, columns=weather_columns)
    meter_df = load_all_meter_series_for_building(building_id, sampling_rate, missingval_tolerance, columns_force=meter_columns)
    return building_id, weather_df.join(meter_df)

def generate_building_dataframe_single_argument(arg):
    return generate_building_dataframe(*arg)

def main():
    sampling_rate = '1h'
    subseries_length = timedelta(hours=512)
    missingval_tolerance = 0.95
    prediction_duration = timedelta(hours=96)
    
    weather_columns = ('airTemperature', 
                       'cloudCoverage', 
                       'dewTemperature', 
                       'precipDepth1HR', 
                       #'precipDepth6HR', 
                       'seaLvlPressure', 
                       'windDirection', 
                       'windSpeed'
                       )
    
    meter_columns = ('electricity', 
                     'chilledwater', 
                     'hotwater', 
                     #'steam', 
                     'gas', 
                     'water', 
                     #'irrigation', 
                     #'solar'
                     )
    
    prediction_variable = 'electricity'
    
    buildings = tuple(list(pd.read_csv(os.path.join(os.path.dirname(__file__),'..',f'data/meters/cleaned/{prediction_variable}_cleaned.csv'),index_col='timestamp').columns.values))
    
    
    
    
    

    with multiprocessing.Pool() as pool:
        args = [(building, sampling_rate, missingval_tolerance, weather_columns, meter_columns) for building in buildings]
        results = pool.map(generate_building_dataframe_single_argument, args)
        
    for building, df in results:
        print(f"Building {building} dataframe:")
        print(df.info())
        # save to csv with decimal precision of 3
        
        
        df.to_csv(os.path.join(os.path.dirname(__file__),'..',f'data/buildings/{building}.csv'), float_format='%.2f')
    
        
        
        
    # for i, (input_batch, output_batch) in enumerate(batches):
    #     if not i%10:
    #         print(f"Batch {i}:")
    #         print(f"Input batch shape: {input_batch.shape}")
    #         print("Input batch head:")
    #         print(input_batch.head())
    #         print("Input batch tail:")
    #         print(input_batch.tail())
    #         print(f"Output batch shape: {output_batch.shape}")
    #         print(f"Output batch head:")
    #         print(output_batch.head())
    #         print(f"Output batch tail:")
    #         print(output_batch.tail())
    #         print()
   
        
        
    


    
    


if __name__=='__main__':
    main()