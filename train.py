import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 超参数
EMBEDDING_DIM = 128  # CNN 输出的 embedding 大小
HIDDEN_DIM = 256  # MLP 隐藏层大小
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.0001

# CNN 编码器
class CNNEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, embedding_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, feature_dim)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, feature_dim, seq_len)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.mean(x, dim=2)  # 全局平均池化
        x = self.fc(x)  # 映射到 embedding
        return x

# MLP Readout
class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPReadout, self).__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 主模型
class SATPredictor(nn.Module):
    def __init__(self, feature_dim, embedding_dim, pred_times):
        super(SATPredictor, self).__init__()
        self.encoder = CNNEncoder(feature_dim, embedding_dim)
        self.readouts = nn.ModuleList([MLPReadout(embedding_dim, feature_dim) for _ in range(pred_times)])

    def forward(self, x):
        embedding = self.encoder(x)
        outputs = [readout(embedding) for readout in self.readouts]
        return outputs

def normalize(y):
    y_min = y.min(dim=1, keepdim=True).values
    y_max = y.max(dim=1, keepdim=True).values
    return (y - y_min) / (y_max - y_min + 1e-8) 

def train_model(model, train_loader, optimizer, criterion, epochs):
    """
    训练模型
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.float(), targets.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            # Loss
            loss = sum(criterion(outputs[i], targets[:, i, :]) for i in range(len(outputs)))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    # 加载数据
    data_file = "./dataset/processed_data.npz"
    data = np.load(data_file)
    x, y = data['x'], data['y']

    # 转换为 PyTorch 张量
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)

    # 创建数据加载器
    dataset = TensorDataset(x_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    feature_dim = x.shape[2]
    pred_times = y.shape[1]
    model = SATPredictor(feature_dim, EMBEDDING_DIM, pred_times)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # 训练模型
    train_model(model, train_loader, optimizer, criterion, EPOCHS)