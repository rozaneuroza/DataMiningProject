import pandas as pd
import numpy as np


#df = pd.read_csv("dataset_mood_smartphone.csv", index_col=0)
#df['time'] = pd.to_datetime(df['time'])
#df_clean = df.copy() 
#print(f"Loaded: {len(df_clean)} rows")


# 1. Remove impossible values
#before = len(df_clean)

# All appCat and screen duration values should be non-negative
#is_duration = df_clean['variable'].str.startswith('appCat') | (df_clean['variable'] == 'screen')
#df_clean = df_clean[~(is_duration & (df_clean['value'] < 0))]

#print(f"Removed {before - len(df_clean)} rows with negative values")


# From investigating value ranges per variable, we see that activity, mood, 
# circumplex.arousal and circumplex.valence are all inside their defined ranges.
# Therefore there are no outliers there to remove.






