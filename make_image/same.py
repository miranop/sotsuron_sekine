# same.py
import dpkt
import socket
from datetime import datetime, timedelta, timezone


def read_packet_dpkt(filepath, time_filter=None, count_limit=None, tz_offset=0):
    """
    dpktでpcapを読み込み、1パケットごとにdictで返す
    timestampはdatetime（tz_offset補正済み）
    """
    from datetime import datetime, timezone, timedelta
    import dpkt
    import socket

    packets = []
    count = 0
    offset = timedelta(hours=tz_offset)  # ← これを追加

    print(f"  dpktでpcap読み込み開始...")

    with open(filepath, 'rb') as f:
        try:
            pcap = dpkt.pcap.Reader(f)
        except:
            f.seek(0)
            pcap = dpkt.pcapng.Reader(f)

        for timestamp, buf in pcap:
            if count_limit and count >= count_limit:
                break

            # UTCからオフセットを適用してローカル時刻に変換
            dt_local = datetime.fromtimestamp(
                timestamp, tz=timezone.utc) + offset
            hour = dt_local.hour

            # time_filterで範囲を制限（オプション）
            if time_filter and not (time_filter[0] <= hour < time_filter[1]):
                continue

            try:
                eth = dpkt.ethernet.Ethernet(buf)
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                ip = eth.data
                src_ip = socket.inet_ntoa(ip.src)
                dst_ip = socket.inet_ntoa(ip.dst)

                packets.append({
                    'timestamp': dt_local,
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'bytes': bytes(ip)
                })

                count += 1
                if count % 100000 == 0:
                    print(f"    読み込み済み: {count:,} パケット")

            except Exception:
                continue

    print(f"  ✓ 読み込み完了: {len(packets):,} パケット")
    return packets


def group_packets_dpkt(packets):
    traffic = {}
    for pkt in packets:
        pair = (pkt["src_ip"], pkt["dst_ip"])
        if pair not in traffic:
            traffic[pair] = []
        traffic[pair].append(pkt)
    return traffic
