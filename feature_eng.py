import numpy as np
import pandas as pd

# Define a function to get a dummy variable based on wheter the data is on-time (1) or late (0)
def label_is_on_time(data, column):
    if data[column] > 0:
        return 0
    else:
        return 1 

# Define a function to set a dummy variable
def label_is_disputed(data, column):
    if data[column] == 'Yes':
        return 1
    else:
        return 0



# Define the function
def label_is_paperless(data, column):
    if data[column] == 'Electronic':
        return 1
    else:
        return 0 
    

# Define the function
def label_is_date_later(data, column_1, column_2):
    if data[column_1] > data[column_2]:
        return 1
    else:
        return 0 
    
    
# Define a function to conver
def convert_to_date(data, *columns):
    for col in columns:
        data[col] = pd.to_datetime(data[col].stack()).unstack()
    
    return data


def get_m_dow(data, column):
    month = data[column].dt.strftime('%m').astype(int)
    day = data[column].dt.strftime('%w').astype(int)
    data = data.drop(column, axis=1)
    
    return data, month, day


# Define a function to get the sin and cos value of the month value
def get_sin_cos_month(data, column):
    sin_col_name = column + '_sin'
    cos_col_name = column + '_cos'
    data[sin_col_name] = np.sin(data[column]*(2.*np.pi/12))
    data[cos_col_name] = np.cos(data[column]*(2.*np.pi/12))
    
    data = data.drop(column, axis=1)
    
    return data


# Define a function to get the sin and cos value of the day of week value
def get_sin_cos_dow(data, column):
    sin_col_name = column + '_sin'
    cos_col_name = column + '_cos'
    data[sin_col_name] = np.sin(data[column]*(2.*np.pi/7))
    data[cos_col_name] = np.cos(data[column]*(2.*np.pi/7))
    
    data = data.drop(column, axis=1)
    
    return data



def convert_date_to_cat(df):
    # Apply the function to create a new column
    df['is_on_time'] = df.apply(label_is_on_time, args=('days_late', ), axis=1)

    # Drop the column
    df = df.drop('days_late', axis=1)
    
    # Apply the function to the column
    df['is_disputed'] = df.apply(label_is_disputed, args=('is_disputed', ), axis=1)
    
    # Create a dummy variable out of is_paperless categorical data
    df['is_paperless'] = df.apply(label_is_paperless, args=('is_paperless', ), axis=1)
    
    # Create a dummy variable out of paperless_date data
    df['is_paperless_later'] = df.apply(label_is_date_later, args=('paperless_date', 'invoice_date'), axis=1)
    
    return df
    
    
    
def convert_cyclical_to_sin_cos(df):
    # Convert the data type
    df = convert_to_date(df,['paperless_date','invoice_date', 'due_date'])
    
    # Get the Year, Month, and Day of Weekk of Paperless Date
    df, df['paperless_month'], df['paperless_dow'] = get_m_dow(df, 'paperless_date')
    
    # Get the Year, Month, and Day of Weekk of Invoice Date
    df, df['invoice_month'], df['invoice_dow'] = get_m_dow(df, 'invoice_date')
    
    # Get the Year, Month, and Day of Weekk of Due Date
    df, df['due_month'], df['due_dow'] = get_m_dow(df, 'due_date')
    
    cyclical_month_columns = [
        'paperless_month',
        'invoice_month',
        'due_month'
        ]

    cyclical_dow_columns = [
        'paperless_dow',
        'invoice_dow',
        'due_dow'
        ]
    
    for i in cyclical_month_columns:
        df = get_sin_cos_month(df, i)
    
    for i in cyclical_dow_columns:
        df = get_sin_cos_dow(df, i)
    
    return df
    
    
    # Define a function to get the dummy data
def get_dummies(data, columns):
    data = pd.get_dummies(data, columns=columns, drop_first=True, dtype=int)
    
    return data






def feature_eng(df):
    df = convert_date_to_cat(df)
    df = get_dummies(df, ['country_code'])
    # df = pd.get_dummies(df, columns=['country_code'], drop_first=True, dtype=int)
    
    # Remove data that will hinder the prediction of future data and/or will leak the data
    columns = ['customer_id', 'invoice_number', 'settled_date', 'days_to_settle']
    df = df.drop(columns, axis=1)
    
    df = convert_cyclical_to_sin_cos(df)
    
    
    return df