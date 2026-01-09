from collections import defaultdict


def group_by_pair(packets):  # 送信元IPアドレスと受け取りのIPアドレスが同じものをまとめる

    group = defaultdict(list)

    for pkt in packets:
        src = pkt["src"]
        dst = pkt["dst"]

        key = (src, dst)
        group[key].append(pkt)

    return group
