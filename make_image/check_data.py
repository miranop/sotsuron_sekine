#!/usr/bin/env python3
"""
inspect_pcap_sample.py

PCAP を全部読み切らずにサンプリングしてファイル全域の時刻レンジを把握するスクリプト。
依存: dpkt（推奨。無ければ scapy にフォールバック）

使い方例:
  python inspect_pcap_sample.py --pcap ./Pcap/Friday-WorkingHours.pcap --sample-interval 100000 --tz-offset -3

説明:
 - sample_interval: N パケットごとに 1 サンプル取得（デフォルト 100000）
 - max_samples: 出力するサンプル上限（デフォルト 200）
 - tz_offset: 時間補正（hours）。例: Atlantic -3h -> -3
"""

import argparse
import datetime
import sys


def utc_to_offset(dt_utc, offset_hours):
    return dt_utc + datetime.timedelta(hours=offset_hours)


def inspect_with_dpkt(path, sample_interval, max_samples, tz_offset):
    try:
        import dpkt
    except Exception:
        return None, "dpkt not available"

    timestamps = []
    total = 0
    try:
        with open(path, "rb") as f:
            pcap = dpkt.pcap.Reader(f)
            for i, (ts, buf) in enumerate(pcap):
                total = i + 1
                if i % sample_interval == 0:
                    timestamps.append(ts)
                    if len(timestamps) >= max_samples:
                        # we still continue counting total but stop storing more samples
                        pass
    except Exception as e:
        return None, f"dpkt read error: {e}"

    return {"total": total, "timestamps": timestamps}, None


def inspect_with_scapy(path, sample_interval, max_samples, tz_offset):
    try:
        from scapy.utils import PcapReader
    except Exception:
        return None, "scapy not available"

    timestamps = []
    total = 0
    try:
        with PcapReader(path) as rdr:
            for i, pkt in enumerate(rdr):
                total = i + 1
                try:
                    ts = float(pkt.time)
                except Exception:
                    continue
                if i % sample_interval == 0:
                    timestamps.append(ts)
                    if len(timestamps) >= max_samples:
                        pass
    except Exception as e:
        return None, f"scapy read error: {e}"

    return {"total": total, "timestamps": timestamps}, None


def format_output(info, tz_offset):
    if not info or info["total"] == 0:
        return "No packets found."

    # convert to datetimes
    dt_list = [datetime.datetime.utcfromtimestamp(
        ts) for ts in info["timestamps"]]
    dt_min = datetime.datetime.utcfromtimestamp(min(info["timestamps"]))
    dt_max = datetime.datetime.utcfromtimestamp(max(info["timestamps"]))

    dt_min_local = utc_to_offset(dt_min, tz_offset)
    dt_max_local = utc_to_offset(dt_max, tz_offset)

    lines = []
    lines.append("=== Sampling result ===")
    lines.append(f"sampled packet count (approx) : {info['total']:,}")
    lines.append(f"[UTC]     {dt_min.isoformat()} 〜 {dt_max.isoformat()}")
    lines.append(
        f"[local tz offset {tz_offset:+}h]  {dt_min_local.isoformat()} 〜 {dt_max_local.isoformat()}")
    lines.append("")
    lines.append("sample timestamps (local):")
    for dt in dt_list:
        lines.append("  • " + utc_to_offset(dt, tz_offset).isoformat())
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pcap", required=True, help="path to pcap file")
    p.add_argument("--sample-interval", type=int, default=100000,
                   help="take 1 sample every N packets (default: 100000)")
    p.add_argument("--max-samples", type=int, default=200,
                   help="maximum number of sample timestamps to keep (default 200)")
    p.add_argument("--tz-offset", type=float, default=0.0,
                   help="hours to add to UTC to get local (e.g., Atlantic -3 => -3)")
    args = p.parse_args()

    path = args.pcap
    sample_interval = max(1, args.sample_interval)
    max_samples = max(1, args.max_samples)
    tz_offset = args.tz_offset

    # try dpkt first (faster), then scapy fallback
    info, err = inspect_with_dpkt(
        path, sample_interval, max_samples, tz_offset)
    backend = "dpkt"
    if err is not None:
        info, err2 = inspect_with_scapy(
            path, sample_interval, max_samples, tz_offset)
        backend = "scapy"
        if err2 is not None:
            print(
                f"Error: neither dpkt nor scapy available or readable.\n dpkt error: {err}\n scapy error: {err2}")
            sys.exit(1)

    print(f"Using backend: {backend}")
    print(format_output(info, tz_offset))


if __name__ == "__main__":
    main()
