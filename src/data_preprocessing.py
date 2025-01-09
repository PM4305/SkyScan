import pandas as pd
import numpy as np

def load_data(economy_path: str, business_path: str) -> pd.DataFrame:
    """
    Load and combine economy and business class flight data.
    
    Args:
        economy_path (str): Path to economy class data CSV
        business_path (str): Path to business class data CSV
    
    Returns:
        pd.DataFrame: Combined dataset
    """
    data1 = pd.read_csv(economy_path)
    data2 = pd.read_csv(business_path)
    
    # Add class labels
    data1['class'] = 'Economy'
    data2['class'] = 'Business'
    
    # Combine datasets
    return pd.concat([data1, data2])

def flight_time(x: int) -> str:
    """
    Categorize flight time into parts of the day.
    
    Args:
        x (int): Hour of the day (0-24)
    
    Returns:
        str: Time category
    """
    if (x > 4) and (x <= 8):
        return "Early Morning"
    elif (x > 8) and (x <= 12):
        return "Morning"
    elif (x > 12) and (x <= 16):
        return "Afternoon"
    elif (x > 16) and (x <= 20):
        return "Evening"
    elif (x > 20) and (x <= 24):
        return "Night"
    else:
        return "Late Night"

def clean_time_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and process time-related columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    df = df.copy()
    
    # Extract hours from arrival and departure times
    df['arrival_hour'] = pd.to_numeric(df['arr_time'].str[:2], errors='coerce').fillna(0).astype(int)
    df['departure_hour'] = pd.to_numeric(df['dep_time'].str[:2], errors='coerce').fillna(0).astype(int)
    
    # Categorize times
    df['arrival_time'] = df['arrival_hour'].apply(flight_time)
    df['departure_time'] = df['departure_hour'].apply(flight_time)
    
    return df

def categorize_stops(x: int) -> str:
    """
    Categorize number of stops.
    
    Args:
        x (int): Number of stops
    
    Returns:
        str: Stop category
    """
    if x == 0:
        return 'zero'
    elif x == 1:
        return 'one'
    else:
        return 'two_or_more'

def clean_stops_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and process stops data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    df = df.copy()
    
    df['stop'] = df['stop'].astype(str)
    df['stop'] = df['stop'].str.replace('non-stop', '0') \
                          .str.replace(' ', '') \
                          .str.replace('stops', '') \
                          .str.replace('\n', '') \
                          .str.replace('\t', '') \
                          .str.replace('-stop', '') \
                          .str.replace('Via', '')
    
    df['stop'] = pd.to_numeric(df['stop'], errors='coerce').fillna(0).astype(int)
    df['stops'] = df['stop'].apply(categorize_stops)
    
    return df

def convert_duration(duration: str) -> float:
    """
    Convert duration string to hours.
    
    Args:
        duration (str): Duration string (e.g., "2h 30m")
    
    Returns:
        float: Duration in hours
    """
    if pd.isnull(duration) or isinstance(duration, (int, float)):
        return float('nan')
    
    if 'h' not in duration and 'm' not in duration:
        return float('nan')
    
    if 'h' in duration and 'm' in duration:
        hours, minutes = duration.split('h ')
        hours = int(float(hours))
        minutes = int(minutes[:-1]) if minutes[:-1] else 0
    elif 'h' in duration:
        hours = int(duration[:-1])
        minutes = 0
    else:
        hours = 0
        minutes = int(duration[:-1])
    
    return round(hours + minutes / 60, 2)

def clean_duration_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and process duration data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    df = df.copy()
    df['duration'] = df['time_taken'].apply(convert_duration)
    return df

def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean price data.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    df = df.copy()
    df['price'] = df['price'].str.replace(',', '', regex=True).astype(float)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete data preprocessing pipeline.
    
    Args:
        df (pd.DataFrame): Raw DataFrame
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    df = clean_time_data(df)
    df = clean_stops_data(df)
    df = clean_duration_data(df)
    df = clean_price_data(df)
    
    # Drop missing values and duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Drop unnecessary columns
    columns_to_drop = ['date', 'ch_code', 'num_code', 'dep_time', 'arr_time', 
                      'stop', 'time_taken', 'arrival_hour', 'departure_hour']
    df = df.drop(columns=columns_to_drop)
    
    # Rename columns
    df = df.rename(columns={'from': 'source_city', 'to': 'destination_city'})
    
    # Reorder columns
    columns_order = ['airline', 'class', 'source_city', 'destination_city', 
                    'departure_time', 'duration', 'stops', 'arrival_time', 
                    'days_left', 'price']
    df = df[columns_order]
    
    return df