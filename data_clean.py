import pandas as pd

def get_df(path):
    df = pd.read_csv(path)
    
    return df


def data_cleaning(df):
    print("Before cleaning:")
    print(df.shape)
    print(df.isna().sum())
    print(df.duplicated().sum())
    
    # TAMBAH LINE UNTUK CLEANING
    
    return df


def rename_columns(df):
    # Define columns name map
    columns_map = {'countryCode':'country_code',
               'customerID':'customer_id',
               'PaperlessDate':'paperless_date',
               'invoiceNumber':'invoice_number',
               'InvoiceDate':'invoice_date',
               'DueDate':'due_date',
               'InvoiceAmount':'invoice_amount',
               'Disputed':'is_disputed',
               'SettledDate':'settled_date',
               'PaperlessBill':'is_paperless',
               'DaysToSettle':'days_to_settle',
               'DaysLate':'days_late'
               }

    # Set the new columns name based on the map
    df.rename(columns = columns_map, inplace = True)
    
    return df