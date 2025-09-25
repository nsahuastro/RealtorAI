import requests
import pandas as pd
import os
import glob

def preprocessing_hdb_dataframe(df):
    """
    Clean HDB dataframe with specific transformations
    """
    # Make a copy to avoid modifying original
    clean_df = df.copy()

    # 1. Convert 'month' to datetime (assumes day=1)
    print("Converting month to datetime...")
    clean_df['month'] = pd.to_datetime(clean_df['month'] + '-01', format='%Y-%m-%d', errors='coerce')

    # 2. Convert lease_commence_date to datetime (assumes January if only year)
    print("Converting lease_commence_date to datetime...")
    clean_df['lease_commence_date'] = pd.to_datetime(clean_df['lease_commence_date'].astype(str) + '-01-01', format='%Y-%m-%d', errors='coerce')
    '''
    def convert_lease_date(date_val):
        if pd.isna(date_val):
            return pd.NaT
        try:
            # If it's just a year (like 1986), assume January 1st
            if len(str(int(date_val))) == 4:
                return pd.to_datetime(f"{int(date_val)}-01-01")
            else:
                return pd.to_datetime(date_val)
        except:
            return pd.NaT
    clean_df['lease_commence_date'] = clean_df['lease_commence_date'].apply(convert_lease_date)
    '''

    #3. Convert remaining_lease to float
    print("Converting remaining_lease to float...")
    def convert_remaining_lease_vectorized(series):
        s = series.fillna('0').astype(str).str.lower()

        # Extract years
        years = s.str.extract(r'(\d+(?:\.\d+)?)\s*year')[0].astype(float).fillna(0)
        # Extract months
        months = s.str.extract(r'(\d+(?:\.\d+)?)\s*month')[0].astype(float).fillna(0)

        # If it's just a number without 'year' or 'month', treat as years
        only_number = s.str.replace(r'[^\d.]', '', regex=True)
        only_number = pd.to_numeric(only_number, errors='coerce').fillna(0)

        # Combine
        total_years = years + months / 12
        total_years = total_years.where(total_years != 0, only_number)  # fallback for plain numbers

        return total_years
    clean_df['remaining_lease'] = convert_remaining_lease_vectorized(clean_df['remaining_lease'])

    # 4. For NaN remaining_lease, calculate from lease_commence_date and month
    print("Calculating missing remaining_lease values...")
    '''
    def calculate_remaining_lease(row):
        if pd.isna(row['remaining_lease']) and pd.notna(row['lease_commence_date']) and pd.notna(row['month']):
            lease_duration = 99
            years_elapsed = (row['month'] - row['lease_commence_date']).days / 365.25
            remaining = lease_duration - years_elapsed
            return max(0, remaining)  # Don't return negative values
        return row['remaining_lease']    
    clean_df['remaining_lease'] = clean_df.apply(calculate_remaining_lease, axis=1)
    '''
    #Fully vectorised faster version
    lease_duration = 99
    mask = clean_df['remaining_lease'].isna() & clean_df['lease_commence_date'].notna() & clean_df['month'].notna()
    years_elapsed = (clean_df.loc[mask, 'month'] - clean_df.loc[mask, 'lease_commence_date']).dt.days / 365.25
    clean_df.loc[mask, 'remaining_lease'] = (lease_duration - years_elapsed).clip(lower=0)

    # 5. Split storey_range into min and max
    print("Splitting storey_range into min and max...")
    # Extract "min TO max" pattern
    ranges = clean_df['storey_range'].str.extract(r'(?P<min>\d+)\s*TO\s*(?P<max>\d+)', expand=True)
    # Extract single numbers if no "TO"
    single = clean_df['storey_range'].str.extract(r'(?P<single>\d+)', expand=True)

    clean_df['storey_range_min'] = ranges['min'].fillna(single['single']).astype(int)
    clean_df['storey_range_max'] = ranges['max'].fillna(single['single']).astype(int)

    return clean_df