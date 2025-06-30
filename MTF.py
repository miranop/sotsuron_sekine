import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pyts.image import MarkovTransitionField
from PIL import Image
import os

df = pd.read_csv('Monday-WorkingHours.pcap_ISCX.csv')
print(f"CSVファイルを読み込みました。")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

#使用する特徴量
feature_columns = [" Flow IAT Mean", " Flow Duration", "Flow Bytes/s"]

# 欠損値の処理
scaled_features = []
for col in feature_columns:
    data = df[col].dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data).flatten()
    scaled_features.append(scaled)

# 最小長に合わせて特徴量を切り詰める
min_len = min(len(f) for f in scaled_features)
scaled_features = [f[:min_len] for f in scaled_features]

window_size = 32
max_image = 100
max_possible = (min_len - window_size)
num_windows = (min_len - window_size) // window_size
split_features = []
for feature in scaled_features:
    windows = [feature[i:i+window_size] for i in range(0, num_windows * window_size, window_size)]
    split_features.append(np.array(windows))

mtf = MarkovTransitionField(image_size=window_size, n_bins=2, strategy='uniform')
mtf_features = [mtf.fit_transform(f) for f in split_features]

output_dir = "rgb_mtf_images2"
os.makedirs(output_dir, exist_ok=True)
for i in range(num_windows):
    r = mtf_features[0][i]
    g = mtf_features[1][i]
    b = mtf_features[2][i]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = ((rgb + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(f"{output_dir}/rgb_mtf_{i:04d}.png")