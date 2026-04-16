import pandas as pd
import numpy as np

# Data Cleaning pipeline 

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
