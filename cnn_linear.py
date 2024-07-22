import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Normalization functions
def normalize(_V):
    return (_V - np.min(_V)) / (np.max(_V) - np.min(_V))

def normalize_joystick_readings(x, y):
    x_magnitude = normalize(x)
    y_magnitude = normalize(y)
    return x_magnitude, y_magnitude

# Load and preprocess data
fname = './data/joystick_track.npz'
alldat = np.load(fname, allow_pickle=True)['dat']
dat = alldat[0]
patient_idx = 0
d = dat[patient_idx]
targetX = d['targetX']
targetY = d['targetY']
x_magnitude, y_magnitude = normalize_joystick_readings(targetX, targetY)
magnitudes = np.stack((x_magnitude, y_magnitude), axis=1)

# Define models
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels, features):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, input_channels//6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d( input_channels//6, input_channels//3, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)#outs 1500
        self.fc1 = nn.Linear(features*input_channels//(6*3*2*5*), 64)
        self.fc2 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        #x = x.view(x.size(0), -1)  # Flatten the tensor
        #print(f"Flattened tensor shape: {x.shape}")  # Debugging line
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PredictionModel(nn.Module):
    def __init__(self, feature_dim):
        super(PredictionModel, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ECoGModel(nn.Module):
    def __init__(self, input_channels, features):
        super(ECoGModel, self).__init__()
        self.feature_extractor = FeatureExtractor(input_channels, features)
        self.prediction_model = PredictionModel(64)

    def forward(self, x):
        features = self.feature_extractor(x)
        predictions = self.prediction_model(features)
        return predictions

# Dataset class
class JoystickDataset(Dataset):
    def __init__(self, ecog_data, magnitudes, batch_size=600):
        # Ensure the data is divisible by 128
        num_samples = (ecog_data.shape[0] // batch_size) * batch_size
        self.ecog_data = ecog_data[:num_samples]
        self.magnitudes = magnitudes[:num_samples // batch_size]

        # Split the data into groups of 128
        self.ecog_data = np.array_split(self.ecog_data, self.ecog_data.shape[0] // batch_size, axis=0)

    def __len__(self):
        return len(self.ecog_data)

    def __getitem__(self, idx):
        x = self.ecog_data[idx]
        y = self.magnitudes[idx]
        #print(f"Data shape: {x.shape}, Magnitude shape: {y.shape}")  # Debugging line
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

if __name__ == '__main__':
    # Load ECoG data
    ecog_data = d['V']
    print(f"Initial ECoG data shape: {ecog_data.shape}")

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Is CUDA available?', torch.cuda.is_available())

    # Create dataset and dataloader
    dataset = JoystickDataset(ecog_data, magnitudes)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Hyperparameters
    print(dataset.ecog_data[0].shape)
    input_channels = dataset.ecog_data[0].shape[0]  # Ensure this is correctly set
    features = dataset.ecog_data[0].shape[1]  # Ensure this is correctly set
    learning_rate = 0.05
    num_epochs = 50

    # Model, loss function, and optimizer
    model = ECoGModel(input_channels, features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        for i, (ecog_data, target_coords) in enumerate(dataloader):
            ecog_data, target_coords = ecog_data.to(device), target_coords.to(device)
            #print(f"Batch {i} ECoG data shape: {ecog_data.shape}")
            model.train()
            optimizer.zero_grad()
            outputs = model(ecog_data)
            
            # Ensure the target and output have the same shape
            target_coords = target_coords.view_as(outputs)
            
            loss = criterion(outputs, target_coords)
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 10 == 0 and i == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training complete.")

    # Save the model
    torch.save(model.state_dict(), 'extractor_prediction_ecog_model.pth')

