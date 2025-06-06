import os
import numpy as np

# 超参数
SEQ_LEN = 50  # 输入序列长度
PRED_TIMES = [10, 20, 50]  # 预测的时间步
SAMPLE_RATIO = 0.1  # 采样比例

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

def prepare_data(data, seq_len, pred_times):
    """
    将数据处理为 x 和 y
    x: [n-50:n] 的矩阵
    y: n+10, n+20, n+50 的特征
    """
    x, y = [], []
    for n in range(seq_len, len(data) - max(pred_times)):
        # 提取 [n-50:n] 的矩阵
        input_seq = data[n-seq_len:n]
        # 提取 n+10, n+20, n+50 的特征
        target_seq = [data[n + t] for t in pred_times]
        x.append(input_seq)
        y.append(target_seq)
    return np.array(x), np.array(y)

def save_to_npz(x, y, output_file):
    """
    将 x 和 y 保存为 .npz 文件
    """
    np.savez_compressed(output_file, x=x, y=y)
    print(f"Data saved to {output_file}")

def process_and_save_individual_files(input_dir, output_dir):
    """
    逐个处理文件夹中的 .log 文件，并保存为单独的 .npz 文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".log"):  # 只处理 .log 文件
            filepath = os.path.join(input_dir, filename)
            print(f"Processing {filepath}...")
            data = load_data(filepath)
            if len(data) == 0:
                print(f"Warning: {filepath} is empty or has no valid data. Skipping.")
                continue
            x, y = prepare_data(data, SEQ_LEN, PRED_TIMES)
            # Random sample x and y based on SAMPLE_RATIO
            sample_size = int(len(x) * SAMPLE_RATIO)
            indices = np.random.choice(len(x), sample_size, replace=False)
            x = x[indices]
            y = y[indices]
            if len(x) == 0 or len(y) == 0:
                print(f"Warning: {filepath} has no valid samples after sampling. Skipping.")
                continue
            # Save to .npz file
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.npz")
            save_to_npz(x, y, output_file)

def merge_npz_files(input_dir, output_file):
    """
    合并文件夹中的所有 .npz 文件为一个最终的 .npz 文件
    """
    all_x, all_y = [], []
    for filename in os.listdir(input_dir):
        if filename.endswith(".npz"):  # 只处理 .npz 文件
            filepath = os.path.join(input_dir, filename)
            print(f"Loading {filepath}...")
            data = np.load(filepath)
            all_x.append(data['x'])
            all_y.append(data['y'])
    # 合并所有数据
    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    # 保存为最终的 .npz 文件
    save_to_npz(x, y, output_file)

if __name__ == "__main__":
    # 输入和输出路径
    input_dir = "dataset/raw_data"
    intermediate_dir = "dataset/individual_npz"
    final_output_file = "dataset/processed_data.npz"

    # 逐个处理并保存每个文件
    process_and_save_individual_files(input_dir, intermediate_dir)

    # 合并所有 .npz 文件
    merge_npz_files(intermediate_dir, final_output_file)