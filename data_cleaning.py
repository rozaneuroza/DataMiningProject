import pandas as pd
import numpy as np

# Task 1: EDA, Data Cleaning & Feature Engineering 

# Step 1: Load data

def load_data(file_path, dtype_dict=None):
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
    return df 

def remove_outliers(df, low=0.01, high=0.99):

    results = []
    variables = df['variable'].unique()

    for var in variables:
        subset = df[df["variable"] == var].copy()

        # Known ranges
        if var == "mood":
            subset = subset[subset["value"].between(1, 10)]

        elif var == "circumplex.arousal":
            subset = subset[subset["value"].between(-2, 2)]

        elif var == "circumplex.valence":
            subset = subset[subset["value"].between(-2, 2)]

        elif var == "activity":
            subset = subset[subset["value"].between(0, 1)]

        else:
            lower = subset["value"].quantile(low)
            upper = subset["value"].quantile(high)
            subset = subset[subset["value"].between(lower, upper)]

        results.append(subset)

    df = pd.concat(results, ignore_index=True)
    return df

# Step 3: Missing values imputation 
    
# ...
# ... 


def save_cleaned(df):
    df.to_csv("df_clean.csv", index = False)

# Step 4: Feature engineering
    
def aggregate_values_daily(df):
    df['time'] = pd.to_datetime(df['time'], format='mixed').dt.normalize()

    SUM_VARS = (
        set(df.loc[df['variable'].str.startswith('appCat'), 'variable'].unique())
        | {'screen', 'call', 'sms'}
    )

    mask_sum  = df['variable'].isin(SUM_VARS)

    daily_sum  = (df[mask_sum]
                .groupby(['id', 'time', 'variable'])['value']
                .sum()
                .reset_index())

    daily_mean = (df[~mask_sum]
                .groupby(['id', 'time', 'variable'])['value']
                .mean()
                .reset_index())

    daily = pd.concat([daily_sum, daily_mean], ignore_index=True)

    daily_pivot = daily.pivot_table(
        index=['id', 'time'],
        columns='variable',
        values='value'
    ).reset_index()

    daily_pivot.columns.name = None

    return daily_pivot

# subpart: Reindex to gain the time series 

def reindex_user(df):
    df = df.sort_values('time').set_index('time')

    # Create full daily date range for this user
    full_range = pd.date_range(df.index.min(), df.index.max(), freq='D')

    # Reindex to insert missing days
    df = df.reindex(full_range)
    df.index.name = 'time'

    return df

def reindex_all_users(df):
    parts = []
    for uid, udf in df.groupby('id'):
        udf = udf.drop(columns='id')
        result = reindex_user(udf)
        result['id'] = uid
        parts.append(result.reset_index())
    return pd.concat(parts, ignore_index=True)


def main():
    # Load data:
    dtype_dict = {
        "id" : "category",
        "variable" : "category",
    }

    df = load_data(file_path = "dataset_mood_smartphone.csv", dtype_dict=dtype_dict)

    # Outliers:
    df = drop_index_column(df)
    df = to_datetime(df)
    df = remove_negatives(df)
    df = remove_outliers(df)

    # Missing values:
    ## .... 
    ## ....

    save_cleaned(df)

    # Feature engineering 
    df = load_data(file_path="df_clean.csv")
    df = aggregate_values_daily(df)
    df = reindex_all_users(df)
    print(df.head(10))




if __name__ == "__main__":
    main()
