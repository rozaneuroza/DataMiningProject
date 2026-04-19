import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data_with_features.csv")
### REGRESSION: RANDOM FOREST ###


print("\n--- RANDOM FOREST REGRESSION ---")

# Features and target
X = df.drop(columns=['id', 'date', 'mood_class', 'mood'])  # remove classification + target
y = df['mood']

# Train-test split
# For temporal data we split it not randomly but by time, to avoid data leakage. 
# We sort by id and date, then take the first 80% as training and the last 20% as testing.
df_sorted = df.sort_values(['id', 'date'])

split_idx = int(0.8 * len(df_sorted))

train_df = df_sorted.iloc[:split_idx]
test_df  = df_sorted.iloc[split_idx:]

X_train = train_df.drop(columns=['id', 'date', 'mood_class', 'mood'])
y_train = train_df['mood']

X_test = test_df.drop(columns=['id', 'date', 'mood_class', 'mood'])
y_test = test_df['mood']

# Model
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf_reg.fit(X_train, y_train)

# Predictions
y_pred = rf_reg.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Mood")
plt.ylabel("Predicted Mood")
plt.title("Random Forest: Actual vs Predicted Mood")

# Perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.tight_layout()
plt.savefig("rf_regression_scatter.png")
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Residual Distribution (Random Forest)")
plt.xlabel("Error (Actual - Predicted)")
plt.tight_layout()
plt.savefig("rf_residuals.png")
plt.show()

importances = rf_reg.feature_importances_
indices = np.argsort(importances)[::-1]
top_10 = indices[:10]

feature_names = X.columns

plt.figure(figsize=(12,6))
plt.title("Top 10 Feature Importances (Regression)")
plt.bar(range(10), importances[top_10])
plt.xticks(range(10), [feature_names[i] for i in top_10], rotation=90)
plt.tight_layout()
plt.savefig("rf_reg_feature_importance.png")
plt.show()
