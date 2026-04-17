import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

USAGE_VARS = ['call', 'sms', 'screen', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather']
STATE_VARS = ["mood", "circumplex.arousal", "circumplex.valence", "activity"]

SECONDS_IN_DAY = 86400

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

def save_cleaned(df):
    df.to_csv("df_clean.csv", index = False)

def save_dataset(df, file_path):
    df.to_csv(file_path, index = False)
    print(f"Saved cleaned dataset to {file_path}")

# Step 4: Feature engineering
    
# Before aggregation - daily variation - compute from daily data
    
def pivot_raw(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['time']).dt.normalize()

    raw_pivot = df.pivot_table(
        index=['id', 'time', 'date'],
        columns='variable',
        values='value'
    ).reset_index()
    
    raw_pivot.columns.name = None
    return raw_pivot


def daily_variation(raw_pivot):
    numeric_columns = raw_pivot.select_dtypes(include="number").columns
    
    new_cols = {}
    for var in numeric_columns:
        new_cols[var] = raw_pivot.groupby(['id', 'date'])[var].transform('std')
    
    variation = pd.DataFrame(new_cols, index=raw_pivot.index)
    variation['id'] = raw_pivot['id']
    variation['date'] = raw_pivot['date']
    
    return variation.groupby(['id', 'date']).first().reset_index()  # ← id and date become plain columns


def aggregate_daily(raw_pivot):
    df = raw_pivot.copy()
    numeric_columns = df.select_dtypes(include="number").columns
    
    agg_dict = {var: ("sum" if var in USAGE_VARS else "mean") for var in numeric_columns}
    
    daily = df.groupby(['id', 'date']).agg(agg_dict).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    
    return daily

def remove_impossible_daily(df):
    duration_cols = [c for c in df.columns if c.startswith('appCat') or c == 'screen']

    impossible_mask = df[duration_cols] > SECONDS_IN_DAY
    df[duration_cols] = df[duration_cols].mask(impossible_mask)

    return df

def reindex_user(user_df):
    user_df = user_df.sort_values('date').set_index('date')

    full_range = pd.date_range(user_df.index.min(), user_df.index.max(), freq='D')
    user_df = user_df.reindex(full_range)
    user_df.index.name = 'date'

    return user_df

def reindex_all_users(df):
    parts = []

    for uid, udf in df.groupby('id'):
        udf = udf.drop(columns='id')
        result = reindex_user(udf)
        result['id'] = uid
        parts.append(result.reset_index())

    return pd.concat(parts, ignore_index=True)

# Missing values

def fill_usage_vars(df):
    usage_cols = [c for c in df.columns if c in USAGE_VARS]
    df[usage_cols] = df[usage_cols].fillna(0)
    return df

def trim_before_first_state(user_df):
    user_df = user_df.sort_values('date').reset_index(drop=True)
    has_state = user_df[STATE_VARS].notna().any(axis=1)
    if not has_state.any():
        return pd.DataFrame(columns=user_df.columns)
    first_valid_pos = has_state.idxmax()
    return user_df.loc[first_valid_pos:].reset_index(drop=True)  # ← fix here

def trim_all_users(df):
    return (
        df.groupby('id', group_keys=False)
        .apply(trim_before_first_state)
        .reset_index(drop=True)
    )

def knn_impute(df):
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    df[STATE_VARS] = imputer.fit_transform(df[STATE_VARS])
    return df

def save_cleaned(df, name):
    df.to_csv(name, index = False)
    print(f"Saved cleaned data to {name}")

# Feature engineering 

# Sliding windows     
def sliding_window_features(df, windows=[3, 7, 14, 30]):
    df = df.sort_values(['id', 'date'])
    numeric_cols = [c for c in df.select_dtypes(include='number').columns if c != 'id']
    
    new_cols = {}
    for var in numeric_cols:
        for w in windows:
            mean_col = f"{var}_mean_{w}d"
            new_cols[mean_col] = (df.groupby('id')[var]
                                    .transform(lambda x: x.rolling(w, min_periods=1).mean()))
            new_cols[f"{var}_diff_{w}d"] = df.groupby('id')[mean_col].diff() if mean_col in df.columns else new_cols[mean_col].groupby(df['id']).diff()
    
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

def main():
    # Load data:

    df = load_data(file_path = "dataset_mood_smartphone.csv")

    # Outliers:
    df = drop_index_column(df)
    df = to_datetime(df)
    df = remove_negatives(df)
    df = remove_outliers(df)

    # Save cleaned dataset before aggregation
    save_cleaned(df, "data_cleaned_before_agg.csv")

    # Daily standard deviation 
    raw_pivot = pivot_raw(df)
    std_daily = daily_variation(raw_pivot)
    # Result - NaNs here mean that either there were no entries, or only one, so the standard deviation doesn't exist
    print(std_daily.head(5))
    save_dataset(std_daily, "daily_standard_variation.csv")

    # Feature engineering
    df = aggregate_daily(raw_pivot)
    df = remove_impossible_daily(df)
    df = reindex_all_users(df)

    # Missing values
    df = fill_usage_vars(df)
    df = trim_all_users(df)
    df = knn_impute(df)

    # Final column order
    cols = ['id', 'date'] + [c for c in df.columns if c not in ('id', 'date')]
    df = df[cols]

    print(df.head(10))

    # Save final cleaned dataset
    save_cleaned(df, "data_cleaned.csv")

    # Feature engineering 

    df = sliding_window_features(df)  

    cols = ['id', 'date'] + [c for c in df.columns if c not in ('id', 'date')]

    print(df.head())

    save_dataset(df, "data_with_features.csv")

if __name__ == "__main__":
    main()
