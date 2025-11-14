#!/usr/bin/env python3
import dpkt
from datetime import datetime

pcap_file = "../Pcap/Wednesday-workingHours.pcap"

with open(pcap_file, 'rb') as f:
    try:
        pcap = dpkt.pcap.Reader(f)
    except ValueError:
        # pcapng形式の場合
        f.seek(0)
        pcap = dpkt.pcapng.Reader(f)

    for i, (timestamp, buf) in enumerate(pcap):
        if i >= 5:
            break

        # ローカルタイム
        dt_local = datetime.fromtimestamp(timestamp)

        # UTC
        dt_utc = datetime.utcfromtimestamp(timestamp)

        print(f"パケット{i}:")
        print(f"  ローカル: {dt_local.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  UTC:      {dt_utc.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  時差:     {(dt_local.hour - dt_utc.hour) % 24} 時間")
        print()
