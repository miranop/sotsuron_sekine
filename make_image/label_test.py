#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
攻撃時間ラベル判定テストスクリプト
- label.py の ATTACK_SCHEDULES を参照し、
  指定したファイルと時刻の判定を確認する
"""

import os
from datetime import datetime, timedelta
from label import ATTACK_SCHEDULES


def get_attack_status(timestamp, pcap_filename):
    """指定時刻が攻撃時間かどうかを判定"""
    base = os.path.basename(pcap_filename).lower()

    # ★修正: utcfromtimestamp → fromtimestamp
    pkt_time = datetime.fromtimestamp(float(timestamp)).time()

    schedules = {k.lower(): v for k, v in ATTACK_SCHEDULES.items()}

    if base not in schedules:
        return "benign", None

    for attack in schedules[base]:
        start = datetime.strptime(attack["start"], "%H:%M:%S").time()
        end = datetime.strptime(attack["end"], "%H:%M:%S").time()
        if start <= pkt_time <= end:
            return "attack", attack["type"]

    return "benign", None


def test_attack_windows(pcap_filename, start_time_str, end_time_str, step_sec=60):
    """
    指定した時間範囲をテスト
    pcap_filename: 例 "Wednesday-workingHours.pcap"
    start_time_str, end_time_str: "09:30:00" の形式
    step_sec: チェック間隔（秒）
    """
    base_date = datetime(2025, 1, 1)
    start_dt = datetime.strptime(start_time_str, "%H:%M:%S").replace(
        year=base_date.year, month=base_date.month, day=base_date.day
    )
    end_dt = datetime.strptime(end_time_str, "%H:%M:%S").replace(
        year=base_date.year, month=base_date.month, day=base_date.day
    )

    cur = start_dt
    print(f"=== {pcap_filename} の {start_time_str}～{end_time_str} のラベル確認 ===\n")

    while cur <= end_dt:
        ts = cur.timestamp()
        status, attack_type = get_attack_status(ts, pcap_filename)
        label = attack_type if status == "attack" else "BENIGN"
        print(f"{cur.strftime('%H:%M:%S')} → {label}")
        cur += timedelta(seconds=step_sec)


if __name__ == "__main__":
    # テスト例
    test_attack_windows(
        pcap_filename="Wednesday-workingHours.pcap",
        start_time_str="09:30:00",
        end_time_str="11:30:00",
        step_sec=120  # 2分おきに確認
    )
