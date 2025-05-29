import numpy as np
import pandas as pd
import PIL.Image
from sklearn.preprocessing import MinMaxScaler



df = pd.read_csv("Monday-WorkingHours.pcap_ISCX.csv", header=0)
print(f"CSVファイルを読み込みました。")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# NaN を含む行を削除
df.dropna(inplace=True)


# --- グループ化 ---
features_r = [  # 赤: パケット/サイズ
    ' Total Fwd Packets', ' Total Backward Packets',
    'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
    ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
    'Bwd Packet Length Max', ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std',
    ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
    'Fwd Packets/s', ' Bwd Packets/s', ' Fwd Header Length', ' Bwd Header Length', ' Fwd Header Length.1',
    'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd', ' min_seg_size_forward',
    ' Down/Up Ratio', ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size',
    'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate',
    ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate'
]

features_g = [  # 緑: 時間/遅延
    ' Flow Duration', 'Flow Bytes/s', ' Flow Packets/s',
    ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
    'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min',
    'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min',
    'Active Mean', ' Active Std', ' Active Max', ' Active Min',
    'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min'
]

features_b = [  # 青: TCPフラグ・接続
    'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags',
    'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
    ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count',
    'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes'
]

#スケーリング
scale_r = MinMaxScaler()
scale_g = MinMaxScaler()
scale_b = MinMaxScaler()

r_scaled = scale_r.fit_transform(df[features_r])
g_scaled = scale_g.fit_transform(df[features_g])
b_scaled = scale_b.fit_transform(df[features_b])

print(np.isinf(df[features_g]).sum())
print(np.isnan(df[features_g]).sum())
print(df[features_g].max())



IMG_SIZE = 12
VECTOR_LEN = IMG_SIZE * IMG_SIZE
#画像生成
for i in range(len(df)):
    def image(vec):
        padded = np.pad(vec, (0, VECTOR_LEN - len(vec)), 'constant')
        norm = (padded - np.min(padded)) / (np.max(padded) - np.min(padded) + 1e-8)
        print("min:", np.min(padded), "max:", np.max(padded))
        return (norm * 255).astype(np.uint8).reshape((IMG_SIZE, IMG_SIZE))
    
r = image(r_scaled[i])
g = image(g_scaled[i])
b = image(b_scaled[i])

rgb = np.stack([r,g,b],axis=1)
img = PIL.Image.fromarray(rgb, mode='RGB')
img.save(f"RGB_{i}.png")
print(f"RGB画像を保存しました: RGB_{i}.png")
