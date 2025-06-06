import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from pathlib import Path
from datetime import datetime

# 超参数
SEQ_LEN = 50  # 输入序列长度
PRED_TIMES = [10, 20, 50]  # 预测的时间步
SAMPLE_RATIO = 0.1  # 采样比例
thread_num = 1

def load_data(filepath):
    """
    加载单个 .log 文件为 numpy 数组
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            features = list(map(float, line.strip().split()))
            data.append(features)
    return np.array(data)

def prepare_data(data, data_scale, seq_len, pred_times):
    """
    将数据处理为 x 和 y
    x: [n-50:n] 的矩阵
    y: n+10, n+20, n+50 的特征
    """
    x, y = [], []
    y_delta = []
    for n in range(seq_len, len(data_scale) - max(pred_times)):
        # 提取 [n-50:n] 的矩阵
        input_seq = data_scale[n-seq_len:n]
        # 提取 n+10, n+20, n+50 的特征
        target_seq = [data_scale[n + t] for t in pred_times]
        change_seq = [0] * len(pred_times)
        for i, t in enumerate(pred_times):
            last_state = data[n]
            future_state = data[n + t]
            change = (future_state - last_state) / (1)  # 避免除以0
            change[change < 0] = -1
            change[change > 0] = 1
            change[change == 0] = 0 
            change_seq[i] = change
            
        x.append(input_seq)
        y.append(target_seq)
        y_delta.append(change_seq)
    return np.array(x), np.array(y), np.array(y_delta)

def save_to_npz(x, y, y_delta, output_file):
    """
    将 x 和 y 保存为 .npz 文件
    """
    np.savez(output_file, x=x, y=y, y_delta=y_delta)
    print(f"Data saved to {output_file}")
    
class SimpleScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit_transform(self, data):
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        self.std_[self.std_ == 0] = 1e-7
        return (data - self.mean_) / self.std_
    
    def transform(self, data):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet. Call fit_transform first.")
        return (data - self.mean_) / self.std_

def process_and_save_individual_file(input_path, otuput_path):
    """
    逐个处理文件夹中的 .log 文件，并保存为单独的 .npz 文件
    """
    if input_path.endswith(".log"):  # 只处理 .log 文件
        print(f"Processing {input_path}...")
        data = load_data(input_path)
        if len(data) == 0:
            print(f"Warning: {input_path} is empty or has no valid data. Skipping.")
            return
        data_scaled = SimpleScaler().fit_transform(data)  # 数据归一化
        x, y, y_delta = prepare_data(data, data_scaled, SEQ_LEN, PRED_TIMES)
        
        # Random sample x and y based on SAMPLE_RATIO
        sample_size = int(len(x) * SAMPLE_RATIO)
        indices = np.random.choice(len(x), sample_size, replace=False)
        x = x[indices]
        y = y[indices]
        y_delta = y_delta[indices]
        if len(x) == 0 or len(y) == 0:
            print(f"Warning: {input_path} has no valid samples after sampling. Skipping.")
            return
        # Save to .npz file
        save_to_npz(x, y, y_delta, otuput_path)
    
def merge_npz_files(input_dir, output_file):
    """
    合并文件夹中的所有 .npz 文件为一个最终的 .npz 文件
    """
    all_x, all_y = [], []
    all_y_delta = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".npz"):  # 只处理 .npz 文件
            filepath = os.path.join(input_dir, filename)
            print(f"Loading {filepath}...")
            data = np.load(filepath)
            all_x.append(data['x'])
            all_y.append(data['y'])
            all_y_delta.append(data['y_delta'])
    # 合并所有数据
    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    y_delta = np.concatenate(all_y_delta, axis=0)
    # 保存为最终的 .npz 文件
    save_to_npz(x, y, y_delta, output_file)

if __name__ == "__main__":
    # 输入和输出路径
    input_dir = "dataset/raw_data"
    intermediate_dir = "dataset/individual_npz"
    final_output_file = "dataset/processed_data.npz"

    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)
    input_file_list = []
    output_file_list = []
    for file_path in os.listdir(input_dir)[:5]:
        input_file_list.append(os.path.join(input_dir, file_path))
        output_file_list.append(os.path.join(intermediate_dir, f"{os.path.splitext(file_path)[0]}.npz"))
    
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = {executor.submit(process_and_save_individual_file, input_file, output_file): input_file for input_file, output_file in zip(input_file_list, output_file_list)}
        for future in as_completed(futures):
            try:
                future.result()  # 获取结果，捕获异常
            except Exception as e:
                print(f"Error processing file {futures[future]}: {e}")
    
    # 合并所有 .npz 文件
    merge_npz_files(intermediate_dir, final_output_file)