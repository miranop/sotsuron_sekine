from collections import defaultdict
import math


def series_per_second(packets):
    traffic = defaultdict(int)

    for pkt in packets:
        ts = pkt["timestamp"]
        size = pkt["size"]

        # timestampの形式を時間を秒単位に変換する
        sec = math.floor(ts)

        # 同じ時間に通信したパケットを全部送信
        traffic[sec] += size

        secs_sorted = sorted(traffic.keys())

        # series を作成
        series = []

        for s in traffic:
            series.append(traffic[s])

    return series, secs_sorted
