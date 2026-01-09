from load_pcap import load_pcap
from flow_util import group_by_ip_pair
from timeseries import to_series_per_second


def main():
    packets = load_pcap("Monday-WorkingHours.pcap")
    flows = group_by_ip_pair(packets)

    for key, pkt_list in flows.items():
        series, secs = to_series_per_second(pkt_list)
        print(key, series[:10])  # 表示だけ


if __name__ == "__main__":
    main()
