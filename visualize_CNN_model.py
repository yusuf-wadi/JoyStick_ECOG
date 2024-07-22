import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from jstick.cnn_linear import JoystickDataset, ECoGModel, DataLoader

def plot_predictions(model, dataloader, num_samples=100):
    model.eval()
    with torch.no_grad():
        fig, ax = plt.subplots()
        for i, (ecog_data, target) in enumerate(dataloader):
            if i >= num_samples:
                break
            ecog_data = ecog_data.to(device)
            predictions = model(ecog_data)
            predictions = predictions.squeeze(0).cpu().numpy()
            target = target.squeeze(0).cpu().numpy()
            print(f"predictions: {(predictions[0], predictions[1])}, target: {(target[0], target[1])}   ")
            ax.plot(predictions[0], predictions[1], 'o', label=f'Predicted {i}')
            ax.plot(target[0], target[1], 'x', label=f'Target {i}')
        ax.set_xlabel('X Magnitude')
        ax.set_ylabel('Y Magnitude')
        ax.set_title('Joystick Predictions (First 100 Samples)')
        plt.show()


def load_jstick_data():
    # Load and preprocess data
    fname = './data/joystick_track.npz'
    alldat = np.load(fname, allow_pickle=True)['dat']
    dat = alldat[0]
    patient_idx = 0
    d = dat[patient_idx]
    # ecog
    ecog_data = d['V'] # Add batch and sequence length dimensions
    # magnitudes
    targetX = d['targetX']
    targetY = d['targetY']
    x_magnitude, y_magnitude = normalize_joystick_readings(targetX, targetY)
    magnitudes = np.stack((x_magnitude, y_magnitude), axis=1)
    
    return ecog_data, magnitudes

def normalize(_V):
    return (_V - np.min(_V)) / (np.max(_V) - np.min(_V))

def normalize_joystick_readings(x, y):
    x_magnitude = normalize(x)
    y_magnitude = normalize(y)
    return x_magnitude, y_magnitude

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load ECoG data random batch sample
    ecog_data, magnitudes = load_jstick_data()
    print(f"Initial ECoG data shape: {ecog_data.shape}")
    print(f"Initial magnitudes shape: {magnitudes.shape}")
    dataset = JoystickDataset(ecog_data, magnitudes)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    input_channels = dataset.ecog_data[0].shape[0]  # Ensure this is correctly set
    features = dataset.ecog_data[0].shape[1]  # Ensure this is correctly set
    model = ECoGModel(input_channels, features).to(device)
    model.load_state_dict(torch.load('./models/extractor_prediction_ecog_model.pth'))
    # plot predictions
    plot_predictions(model, dataloader)
