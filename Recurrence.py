from collections import defaultdict
from matplotlib import pyplot as plt
from pyts.image import RecurrencePlot
import math
import pyshark
import numpy as np

cap = pyshark.FileCapture('Monday-WorkingHours.pcap')

time_stamps = [] #後で時系列にするためのパケット到着時間格納用
sizes = [] #各パケットのサイズを格納

MAX_PACKETS = 10000  # 処理したい最大件数

for i, pkt in enumerate(cap):
    if i >= MAX_PACKETS:
        break  # 件数上限に達したらループ終了

    try:
        time_stamps.append(float(pkt.sniff_time.timestamp()))
        sizes.append(int(pkt.length))
    except:
        continue  # 欠損値がある場合はスキップ

ts_rounded = [math.floor(t) for t in time_stamps]#秒単位に丸める
traffic_per_sec = defaultdict(int)#いつもの辞書

for t, s in zip(ts_rounded, sizes):
    traffic_per_sec[t] += s

sorted_times = sorted(traffic_per_sec.keys())#時系列にソート
traffic_values = [traffic_per_sec[t] for t in sorted_times]

rp = RecurrencePlot()
X_rp = rp.fit_transform(np.array(traffic_values).reshape(1, -1))

plt.imshow(X_rp[0], cmap='binary')
plt.title("Recurrence Plot from Pcap")
plt.savefig("recurrence_plot.png")
print("✅ recurrence_plot.png に画像を保存しました")
