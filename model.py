import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import Tuple, Optional
import pickle


class TradingLSTM(nn.Module):
    """
    LSTM-based neural network for trading signal prediction
    Combines M3 and H1 timeframes for improved forecasting
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 3):
        super(TradingLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers for processing sequential data
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Activation function
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def train_model(X_train: np.ndarray, y_train: np.ndarray, 
                model_save_path: str = "models/model.pth",
                metadata_save_path: str = "models/metadata.pkl") -> None:
    """
    Trains the neural network model
    
    Args:
        X_train: Training features
        y_train: Training targets
        model_save_path: Path to save the trained model
        metadata_save_path: Path to save training metadata
    """
    # TODO: implement actual training logic
    print("Training model...")
    print(f"Input shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    
    # Determine input size from the training data
    input_size = X_train.shape[1]
    
    # Initialize the model
    model = TradingLSTM(input_size=input_size)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Convert data to PyTorch tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    
    # Add sequence dimension (for LSTM)
    X_tensor = X_tensor.unsqueeze(1)  # Shape: (batch_size, 1, features)
    
    # Train the model (simplified training loop)
    model.train()
    for epoch in range(10):  # TODO: adjust epochs as needed
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
    
    # Save the trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    
    # Save metadata
    metadata = {
        'input_size': input_size,
        'training_date': str(np.datetime64('now')),
        'train_samples': len(X_train),
        'train_accuracy': 0.0  # TODO: calculate actual accuracy
    }
    
    with open(metadata_save_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Model saved to {model_save_path}")
    print(f"Metadata saved to {metadata_save_path}")


def load_model(model_path: str = "models/model.pth", 
               metadata_path: str = "models/metadata.pkl") -> Tuple[nn.Module, dict]:
    """
    Loads a pre-trained model and its metadata
    
    Args:
        model_path: Path to the saved model
        metadata_path: Path to the saved metadata
        
    Returns:
        Tuple of (model, metadata)
    """
    # Load metadata first to get input size
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Initialize model with the same architecture
    model = TradingLSTM(input_size=metadata['input_size'])
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Set model to evaluation mode
    model.eval()
    
    return model, metadata


def predict_direction(model: nn.Module, features: np.ndarray) -> Tuple[int, float, float]:
    """
    Makes a prediction using the trained model
    
    Args:
        model: Trained model
        features: Input features for prediction
        
    Returns:
        Tuple of (signal, probability, confidence)
    """
    # Convert features to tensor and add batch dimension
    features_tensor = torch.FloatTensor(features).unsqueeze(0)
    
    # Add sequence dimension for LSTM (batch_size, seq_len, features)
    features_tensor = features_tensor.unsqueeze(1)
    
    # Make prediction
    with torch.no_grad():
        model.eval()
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get the predicted class and its probability
        predicted_class = torch.argmax(probabilities, dim=1).item()
        max_prob = torch.max(probabilities, dim=1)[0].item()
        
        # Convert class index to signal (-1, 0, 1)
        if predicted_class == 0:
            signal = -1  # Down
        elif predicted_class == 1:
            signal = 0   # Neutral
        else:
            signal = 1   # Up
    
    return signal, max_prob, max_prob