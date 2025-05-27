import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gc

class SimpleScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit_transform(self, data):
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        self.std_[self.std_ == 0] = 1
        return (data - self.mean_) / self.std_
    
    def inverse_transform(self, data):
        return data * self.std_ + self.mean_

class LogDataset(Dataset):
    def __init__(self, data, n_history=10, predict_steps=[1, 2, 3]):
        self.data = data
        self.n_history = n_history
        self.predict_steps = predict_steps
        self.max_step = max(predict_steps)
        self.X, self.y = self._create_sequences()
    
    def _create_sequences(self):
        X, y = [], []
        for i in range(self.n_history, len(self.data) - self.max_step):
            x_seq = self.data[i-self.n_history:i]
            y_seq = []
            for step in self.predict_steps:
                y_seq.append(self.data[i + step - 1])
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
        
        self.relu = nn.ReLU()
        
        # Reduced model complexity
        self.conv1 = nn.Conv1d(input_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(2)
        
        self.dropout_conv = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.3)
        
        conv_output_size = self._get_conv_output_size()
        
        # Smaller fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, predict_steps * input_features)
        
    def _get_conv_output_size(self):
        dummy_input = torch.randn(1, self.input_features, self.n_history)
        x = self.relu(self.bn1(self.conv1(dummy_input)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        return x.numel()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)
        
        x = x.view(-1, self.predict_steps, self.input_features)
        
        return x

def load_log_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            row = [float(x) for x in line.strip().split()]
            data.append(row)
    return np.array(data)

def preprocess_data(data):
    scaler = SimpleScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.7, min_lr=1e-6)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            # Clear memory after each batch
            del outputs, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                # Clear memory after each batch
                del outputs, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        print(f'Best Val Loss: {best_val_loss:.6f}, Patience: {patience_counter}/{patience}')
        
        # Clear memory after each epoch
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}!')
            print(f'Best validation loss: {best_val_loss:.6f}')
            model.load_state_dict(best_model_state)
            break
    
    return train_losses, val_losses

def predict_future(model, input_sequence, scaler, device):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)
        prediction = model(input_tensor)
        prediction_np = prediction.cpu().numpy().reshape(-1, input_sequence.shape[1])
        prediction_original = scaler.inverse_transform(prediction_np)
        return prediction_original

def main():
    # Reduced parameters
    n_history = 50  # Reduced from 100
    predict_steps = [10, 20, 50]
    batch_size = 16  # Reduced from 64
    epochs = 500
    train_ratio = 0.8
    learning_rate = 0.00001
    
    print("Loading data...")
    data = load_log_data('a1.log')
    print(f"Data shape: {data.shape}")
    
    print("Preprocessing data...")
    data_scaled, scaler = preprocess_data(data)
    
    dataset = LogDataset(data_scaled, n_history=n_history, predict_steps=predict_steps)
    print(f"Dataset size: {len(dataset)}")
    
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    print(f"Training set: first {train_size} samples (temporal order)")
    print(f"Validation set: last {val_size} samples (temporal order)")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    input_features = data.shape[1]
    model = LogCNN(input_features=input_features, 
                   n_history=n_history, 
                   predict_steps=len(predict_steps))
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
    
    print("Starting training with regularization...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                         epochs=epochs, lr=learning_rate)
    
    torch.save(model.state_dict(), 'log_cnn_model.pth')
    print("Model saved as log_cnn_model.pth")
    
    np.save('train_losses.npy', train_losses)
    np.save('val_losses.npy', val_losses)
    print("Loss history saved as train_losses.npy and val_losses.npy")
    
    print("\nMaking example prediction...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
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