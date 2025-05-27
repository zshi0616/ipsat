import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SimpleScaler:
    """Simple standardization scaler using numpy"""
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit_transform(self, data):
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1
        return (data - self.mean_) / self.std_
    
    def inverse_transform(self, data):
        return data * self.std_ + self.mean_

class LogDataset(Dataset):
    def __init__(self, data, n_history=10, predict_steps=[1, 2, 3]):
        """
        data: log data, each row is a time step feature
        n_history: how many historical rows to use for prediction
        predict_steps: predict T+X, T+Y, T+Z rows, default is [1,2,3]
        """
        self.data = data
        self.n_history = n_history
        self.predict_steps = predict_steps
        self.max_step = max(predict_steps)
        
        # Create input-output pairs
        self.X, self.y = self._create_sequences()
    
    def _create_sequences(self):
        X, y = [], []
        
        # Ensure enough data to create sequences
        for i in range(self.n_history, len(self.data) - self.max_step):
            # Input: historical n rows of data
            x_seq = self.data[i-self.n_history:i]
            
            # Output: future specified steps of data
            y_seq = []
            for step in self.predict_steps:
                y_seq.append(self.data[i + step - 1])  # step-1 because index starts from 0
            
            X.append(x_seq)
            y.append(np.array(y_seq))
        
        return np.array(X), np.array(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

class LogCNN(nn.Module):
    def __init__(self, input_features, n_history=10, predict_steps=3):
        super(LogCNN, self).__init__()
        
        self.input_features = input_features
        self.n_history = n_history
        self.predict_steps = predict_steps
        
        # Activation function (define early)
        self.relu = nn.ReLU()
        
        # Reduced model complexity to prevent overfitting
        # 1D convolutional layers (smaller channels)
        self.conv1 = nn.Conv1d(input_features, 32, kernel_size=3, padding=1)  # Reduced from 64
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)              # Reduced from 128
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)             # Reduced from 256
        
        # Batch normalization for better training
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool1d(2)
        
        # Increased dropout for regularization
        self.dropout_conv = nn.Dropout(0.3)  # For conv layers
        self.dropout_fc = nn.Dropout(0.5)    # For FC layers
        
        # Calculate input dimension for fully connected layer
        conv_output_size = self._get_conv_output_size()
        
        # Smaller fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)  # Reduced from 512
        self.fc2 = nn.Linear(256, 128)               # Reduced from 256
        self.fc3 = nn.Linear(128, predict_steps * input_features)
        
    def _get_conv_output_size(self):
        # Create a dummy input to calculate size after convolution
        dummy_input = torch.randn(1, self.input_features, self.n_history)
        x = self.relu(self.bn1(self.conv1(dummy_input)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        return x.numel()
    
    def forward(self, x):
        # x shape: (batch_size, n_history, input_features)
        # Convert to (batch_size, input_features, n_history) for 1D convolution
        x = x.transpose(1, 2)
        
        # Convolutional layers with batch norm and dropout
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with higher dropout
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)
        
        # Reshape output to (batch_size, predict_steps, input_features)
        x = x.view(-1, self.predict_steps, self.input_features)
        
        return x

def load_log_data(file_path):
    """Load log data"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # Split each line's numbers and convert to float
            row = [float(x) for x in line.strip().split()]
            data.append(row)
    return np.array(data)

def preprocess_data(data):
    """Data preprocessing"""
    # Standardization
    scaler = SimpleScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    """Train model with early stopping and regularization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    # Add weight decay for L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 20  # Stop if no improvement for 20 epochs
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # if epoch % 10 == 0:
        print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        print(f'Best Val Loss: {best_val_loss:.6f}, Patience: {patience_counter}/{patience}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}!')
            print(f'Best validation loss: {best_val_loss:.6f}')
            # Load best model
            model.load_state_dict(best_model_state)
            break
    
    return train_losses, val_losses

def predict_future(model, input_sequence, scaler, device):
    """Predict future data"""
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)
        prediction = model(input_tensor)
        
        # Inverse standardization
        prediction_np = prediction.cpu().numpy().reshape(-1, input_sequence.shape[1])
        prediction_original = scaler.inverse_transform(prediction_np)
        
        return prediction_original

def main():
    # Parameter settings
    n_history = 100  # Use past 10 rows of data
    predict_steps = [10, 20, 50]  # Predict T+1, T+2, T+3 rows
    batch_size = 64  # Increased batch size for better generalization
    epochs = 200
    train_ratio = 0.8
    learning_rate = 0.00001  # Reduced learning rate for stability
    
    # Load data
    print("Loading data...")
    data = load_log_data('ad14.log')
    print(f"Data shape: {data.shape}")
    
    # Data preprocessing
    print("Preprocessing data...")
    data_scaled, scaler = preprocess_data(data)
    
    # Create dataset
    dataset = LogDataset(data_scaled, n_history=n_history, predict_steps=predict_steps)
    print(f"Dataset size: {len(dataset)}")
    
    # Split training and validation sets by temporal order (not random)
    # This is more appropriate for time series data
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    # Create indices for temporal split
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    
    # Create subsets using temporal order
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    print(f"Training set: first {train_size} samples (temporal order)")
    print(f"Validation set: last {val_size} samples (temporal order)")
    
    # Create data loaders
    # Note: shuffle=False for validation to maintain temporal order
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_features = data.shape[1]
    model = LogCNN(input_features=input_features, 
                   n_history=n_history, 
                   predict_steps=len(predict_steps))
    
    # Calculate and display parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
    
    # Train model
    print("Starting training with regularization...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                         epochs=epochs, lr=learning_rate)
    
    # Save model
    torch.save(model.state_dict(), 'log_cnn_model.pth')
    print("Model saved as log_cnn_model.pth")
    
    # Save loss history
    np.save('train_losses.npy', train_losses)
    np.save('val_losses.npy', val_losses)
    print("Loss history saved as train_losses.npy and val_losses.npy")
    
    # Example prediction
    print("\nMaking example prediction...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Use last n_history rows for prediction
    last_sequence = data_scaled[-n_history:]
    prediction = predict_future(model, last_sequence, scaler, device)
    
    print("Prediction results:")
    for i, step in enumerate(predict_steps):
        print(f"T+{step} row prediction: {prediction[i]}")
    
    print("\nActual last few rows (for comparison):")
    for i in range(min(3, len(data))):
        print(f"Last {3-i} row: {data[-(3-i)]}")

if __name__ == "__main__":
    main()