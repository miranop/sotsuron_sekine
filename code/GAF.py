import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField
from PIL import Image 


# CSVファイルの読み込み
df = pd.read_csv("Monday-WorkingHours.pcap_ISCX.csv", header=0)
print(f"CSVファイルを読み込みました。")
df.replace([np.inf, -np.inf], np.nan, inplace=True)


#使用する特徴量
feature_columns = [" Flow IAT Mean", " Flow Duration", "Flow Bytes/s"]

scaled_features = []
for col in feature_columns:
    data = df[col].dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data).flatten()
    scaled_features.append(scaled)

min_len = min(len(f) for f in scaled_features)
scaled_features = [f[:min_len] for f in scaled_features]

window_size = 32
max_image = 100
max_possible = (min_len - window_size)
num_windows = min(max_image, max_possible)
split_features = []
for feature in scaled_features:
    windows = [feature[i:i+window_size] for i in range(0, num_windows * window_size, window_size)]
    split_features.append(np.array(windows))

gaf = GramianAngularField(image_size=window_size, method='difference')
gaf_features = [gaf.fit_transform(f) for f in split_features]

output_dir = "rgb_gaf_images2"
os.makedirs(output_dir, exist_ok=True)
for i in range(num_windows):
    r = gaf_features[0][i]
    g = gaf_features[1][i]
    b = gaf_features[2][i]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = ((rgb + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(f"{output_dir}/rgb_gaf_{i:04d}.png")
    

