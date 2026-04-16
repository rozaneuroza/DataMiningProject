import pandas as pd
import numpy as np

# Task 1: EDA, Data Cleaning & Feature Engineering 

# Step 1: Load data

def load_data(file_path, dtype_dict):
    df = pd.read_csv(file_path, dtype = dtype_dict, low_memory=False)
    return df

def drop_index_column(df):
    df = df[df.columns[1:]]
    return df 

def to_datetime(df):
    df['time'] = pd.to_datetime(df['time'])
    return df

# Step 2: Remove negative values and outliers/impossible values 

def remove_negatives(df):
    is_duration = df['variable'].str.startswith('appCat') | (df['variable'] == 'screen')
    df = df[~(is_duration & (df['value'] < 0))]

def remove_outliers(df, low = 0.01, high = 0.99):
    # find extreme / impossible values for each category
    results = [] 
    variables = df['variable'].unique()

    for var in variables:
        subset = df[df["variable"] == var]["value"]

        # Variables with known ranges
        if var == "mood":
            subset = subset[subset["value"]].between(1, 10)

        elif var == "circumplex.arousal":
            subset = subset[subset["value"]].between(-2, 2)
        
        elif var == "circumplex.valence":
            subset = subset[subset["value"]].between(-2, 2)

        elif var == "activity":
            subset = subset[subset["value"]].between(0, 1)

        else: 
            # Variables we don't know ranges for 
            lower = subset.quantile(low)
            higher = subset.quantile(high)
            subset = subset[subset["value"].between(lower, higher)]
        
        results.append(subset)

    df = pd.concat(results)
    return df 

# Step 3: Reindex to gain the time series 

def reindex_user(user_df):
    user_df = user_df.sort_values('date').set_index('date')

    # Create full daily date range for this user
    full_range = pd.date_range(user_df.index.min(), user_df.index.max(), freq='D')

    # Reindex to insert missing days
    user_df = user_df.reindex(full_range)
    user_df.index.name = 'date'

    return user_df

def save_cleaned(df):
    df.to_csv("df_clean.csv", index = False)

# Step 4: Missing values imputation 
    

# Step 5: Feature engineering
    
def aggregate_values_daily(df):
    SUM_VARS = (
        set(df.loc[df['variable'].str.startswith('appCat'), 'variable'].unique())
        | {'screen', 'call', 'sms'}
    )

    mask_sum  = df['variable'].isin(SUM_VARS)

    daily_sum  = (df[mask_sum]
                .groupby(['id', 'date', 'variable'])['value']
                .sum()
                .reset_index())

    daily_mean = (df[~mask_sum]
                .groupby(['id', 'date', 'variable'])['value']
                .mean()
                .reset_index())

    daily = pd.concat([daily_sum, daily_mean], ignore_index=True)

    daily_pivot = daily.pivot_table(
        index=['id', 'date'],
        columns='variable',
        values='value'
    ).reset_index()

    daily_pivot.columns.name = None
    daily_pivot['date'] = pd.to_datetime(daily_pivot['date'])

    return daily_pivot

def main():

    dtype_dict = {
        "id" : "category",
        "variable" : "category",
        "value" : "int"
    }

    df = load_data(file_path = "dataset_mood_smartphone.csv", dtype_dict=dtype_dict)

    df = drop_index_column(df)
    df = to_datetime(df)
    df = remove_negatives(df)
    df = remove_outliers(df)
    df = reindex_user(df)
    save_cleaned(df)



if __name__ == "__main__":
    main()
