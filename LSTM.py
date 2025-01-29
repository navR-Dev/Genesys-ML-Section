import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 1. Import and preprocess the data
# Load the dataset
data = pd.read_csv('household_power_consumption.txt', sep=';', na_values=['?'])

# Handle missing values (if any)
data.dropna(inplace=True)

# Combine Date and Time into a single datetime column
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')

# Extract features and target variable
X = data[['Global_active_power', 'Voltage', 'Sub_metering_1']].values
y = data['Global_intensity'].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to torch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 2. Split the data into training and testing sets
train_size = int(0.8 * len(X_tensor))  # 80% for training
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# 3. Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, seq_length, input_size]
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Only take the output of the last time step
        return out

# Initialize the model
input_size = 3  # Number of features (Global_active_power, Voltage, Sub_metering_1)
hidden_size = 64  # Number of LSTM hidden units
output_size = 1  # Predicting Global_intensity

model = LSTMModel(input_size, hidden_size, output_size)

# 4. Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Batch Training Parameters
batch_size = 64  # Define batch size
num_epochs = 10  # Number of epochs

# 6. Train the Model with Mini-Batch Training
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0
    num_batches = len(X_train) // batch_size  # Number of batches

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size

        # Get the current batch
        batch_X = X_train[start_idx:end_idx].unsqueeze(1)  # Adding sequence dimension
        batch_y = y_train[start_idx:end_idx]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)  # Get the model's prediction
        loss = criterion(outputs, batch_y)  # Calculate the loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update the epoch loss
        epoch_loss += loss.item()

    # Print the loss at the end of each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / num_batches:.4f}')

# 7. Evaluate the Model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predictions = model(X_test.unsqueeze(1))  # Predict on the test set
    test_loss = criterion(predictions, y_test)  # Calculate test loss

print(f'Test Loss: {test_loss.item():.4f}')
