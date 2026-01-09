from collections import defaultdict
import socket
import dpkt


def load_pcap(filepath):  # pcapファイルの読み取り&イーサネットに分ける
    packets = []
    with open(filepath, "rb") as f:
        pcap = dpkt.pcap.Reader((f, 'rb'))

        for timestamp, buf in pcap:
            try:
                ethernet = dpkt.ethernet.Ethernet(buf)
                ip = ethernet.data

                if not isinstance(ip, dpkt.ip.IP):
                    continue
                src = socket.inet_ntoa(ip.src)
                dst = socket.inet_ntoa(ip.dst)

                packets.append({
                    "timestamp": timestamp,
                    "src": src,
                    "dst": dst,
                    "bytes": buf,
                    "size": len(buf),
                })
            except Exception:
                # 壊れているものは飛ばす
                continue
    return packets
