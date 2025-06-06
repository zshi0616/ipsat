import torch 
import torch.nn as nn

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
    
# 1D CNN 编码器
class CNN1DEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(CNN1DEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1)  # 1D 卷积，kernel_size=1
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)  # 再次使用 1D 卷积
        self.fc = nn.Linear(128, embedding_dim)  # 映射到 embedding_dim
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, feature_dim)
        x = x[:, -1, :]  # 取最后一个时间步，形状变为 (batch_size, input_dim)
        x = x.unsqueeze(2)  # 添加一个维度，变为 (batch_size, input_dim, 1)
        x = self.relu(self.conv1(x))  # 通过第一个卷积层
        x = self.relu(self.conv2(x))  # 通过第二个卷积层
        x = x.squeeze(2)  # 去掉最后一个维度，变为 (batch_size, 128)
        x = self.fc(x)  # 映射到 embedding_dim
        return x
    
class TopModel(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(TopModel, self).__init__()
        self.cnn_encoder = CNNEncoder(input_dim, embedding_dim)
        self.cnn1d_encoder = CNN1DEncoder(input_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)  # 用于处理残差连接后的输出

    def forward(self, x):
        # CNNEncoder 处理整个序列
        cnn_encoder_output = self.cnn_encoder(x)  # 输出形状: (batch_size, embedding_dim)

        # CNN1DEncoder 处理最后一个时间步
        cnn1d_encoder_output = self.cnn1d_encoder(x)  # 输出形状: (batch_size, embedding_dim)

        # 残差连接
        combined_representation = cnn_encoder_output + cnn1d_encoder_output

        # 进一步处理
        output = self.fc(combined_representation)  # 输出形状: (batch_size, embedding_dim)
        return output