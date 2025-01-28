import warnings
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# Load data
file_path = 'household_power_consumption.txt'
df = pd.read_csv(file_path, sep=';', parse_dates=[[0, 1]], infer_datetime_format=True, na_values=['?'])
df.rename(columns={'Date_Time': 'timestamp'}, inplace=True)

# Convert target column to numeric and drop missing values
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
df.dropna(inplace=True)

# Set timestamp as index and resample to hourly data
df.set_index('timestamp', inplace=True)
df = df.resample('H').mean()

# Add lag and time-based features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['lag_1'] = df['Global_active_power'].shift(1)
df['lag_24'] = df['Global_active_power'].shift(24)  # Lag of 1 day

# Drop missing values caused by lagging
df.dropna(inplace=True)

# Features and target
X = df[['lag_1', 'lag_24', 'hour', 'day_of_week']]
y = df['Global_active_power']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Random Forest MAE: {mae:.3f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:100], label='Actual', color='blue')
plt.plot(y_pred[:100], label='Predicted', color='orange')
plt.title('Random Forest: Actual vs Predicted')
plt.legend()
plt.show()
