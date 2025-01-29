import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Load and Preprocess the Dataset
data = pd.read_csv('household_power_consumption.txt', sep=';', na_values=['?'])

# Handle missing values
data.dropna(inplace=True)

# Combine Date and Time into a single datetime column
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], dayfirst=True)

# Drop original Date and Time columns
data.drop(columns=['Date', 'Time'], inplace=True)

# Convert all numeric columns to proper types
data = data.apply(pd.to_numeric, errors='coerce')

# Drop any remaining NaN values
data.dropna(inplace=True)

# Step 2: Select Features and Target Variable
X = data[['Global_active_power', 'Voltage', 'Sub_metering_1']].values
y = data['Global_intensity'].values

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define Membership Functions Using skfuzzy
# Define universe for input variables
global_active_power_universe = np.arange(0, 8, 0.1)
voltage_universe = np.arange(220, 250, 0.1)
sub_metering_1_universe = np.arange(0, 50, 0.1)

# Define fuzzy membership functions (triangular)
global_active_power = {
    'low': fuzz.trimf(global_active_power_universe, [0, 0, 4]),
    'medium': fuzz.trimf(global_active_power_universe, [2, 4, 6]),
    'high': fuzz.trimf(global_active_power_universe, [4, 8, 8])
}

voltage = {
    'low': fuzz.trimf(voltage_universe, [220, 220, 235]),
    'medium': fuzz.trimf(voltage_universe, [230, 235, 240]),
    'high': fuzz.trimf(voltage_universe, [235, 250, 250])
}

sub_metering_1 = {
    'low': fuzz.trimf(sub_metering_1_universe, [0, 0, 25]),
    'medium': fuzz.trimf(sub_metering_1_universe, [10, 25, 40]),
    'high': fuzz.trimf(sub_metering_1_universe, [25, 50, 50])
}

# Define output membership functions for global_intensity
global_intensity_universe = np.arange(0, 20, 0.1)
global_intensity = {
    'low': fuzz.trimf(global_intensity_universe, [0, 0, 10]),
    'medium': fuzz.trimf(global_intensity_universe, [5, 10, 15]),
    'high': fuzz.trimf(global_intensity_universe, [10, 20, 20])
}

# Step 5: Define Fuzzy Rules
def apply_fuzzy_rules(input_data):
    # Extract features
    global_active_power_val, voltage_val, sub_metering_1_val = input_data

    # Fuzzify the inputs
    global_active_power_fuzz = {
        'low': fuzz.interp_membership(global_active_power_universe, global_active_power['low'], global_active_power_val),
        'medium': fuzz.interp_membership(global_active_power_universe, global_active_power['medium'], global_active_power_val),
        'high': fuzz.interp_membership(global_active_power_universe, global_active_power['high'], global_active_power_val)
    }

    voltage_fuzz = {
        'low': fuzz.interp_membership(voltage_universe, voltage['low'], voltage_val),
        'medium': fuzz.interp_membership(voltage_universe, voltage['medium'], voltage_val),
        'high': fuzz.interp_membership(voltage_universe, voltage['high'], voltage_val)
    }

    sub_metering_1_fuzz = {
        'low': fuzz.interp_membership(sub_metering_1_universe, sub_metering_1['low'], sub_metering_1_val),
        'medium': fuzz.interp_membership(sub_metering_1_universe, sub_metering_1['medium'], sub_metering_1_val),
        'high': fuzz.interp_membership(sub_metering_1_universe, sub_metering_1['high'], sub_metering_1_val)
    }

    # Fuzzy rules
    rule1 = np.fmin(np.fmin(global_active_power_fuzz['low'], voltage_fuzz['low']), sub_metering_1_fuzz['low'])
    rule2 = np.fmin(np.fmin(global_active_power_fuzz['medium'], voltage_fuzz['medium']), sub_metering_1_fuzz['medium'])
    rule3 = np.fmin(np.fmin(global_active_power_fuzz['high'], voltage_fuzz['high']), sub_metering_1_fuzz['high'])

    # Apply rules to output
    output_low = np.fmin(rule1, global_intensity['low'])
    output_medium = np.fmin(rule2, global_intensity['medium'])
    output_high = np.fmin(rule3, global_intensity['high'])

    # Defuzzification: centroid method
    global_intensity_output = fuzz.defuzz(global_intensity_universe, np.fmax(output_low, np.fmax(output_medium, output_high)), 'centroid')

    return global_intensity_output

# Step 6: Apply Fuzzy Inference to All Training Data
y_pred_train = [apply_fuzzy_rules(sample) for sample in X_train]

# Step 7: Evaluate the Model
mse = mean_squared_error(y_train, y_pred_train)
print(f'Mean Squared Error (Train Data): {mse}')

# Step 8: Apply the Model to Test Data
y_pred_test = [apply_fuzzy_rules(sample) for sample in X_test]

# Step 9: Evaluate on Test Data
test_mse = mean_squared_error(y_test, y_pred_test)
print(f'Mean Squared Error (Test Data): {test_mse}')
