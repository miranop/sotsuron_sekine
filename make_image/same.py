#処理の共通部分を記しておく
from scapy.all import rdpcap, IP

def read_packet(filepath,count=1000):#パケットを
    try:
        ip_packets = []#分けたIPアドレスの格納用
        packets = rdpcap(filepath,count=count)
    except FileNotFoundError:
        print(f"Error: pcap file not found at '{filepath}'")
        return []
    
    
    for packet in packets:#読み込んだパケットを見る
        if IP in packet:#パケットの中のIPだけ抜き出す
            ip_packets.append(packet)#追加
    return ip_packets

def group_packets(ip_packets):
    #グループ化
    #グループ格納のための辞書を用意
    traffic = {}
    
    for packet in ip_packets:
        src = packet[IP].src
        dst = packet[IP].dst
        
        #送信元と宛先のIPをタプルにして管理
        ip_pair = (src,dst)
        
        if ip_pair not in traffic:
            traffic[ip_pair] = []
        
        # 該当するIPペアのリストにパケットを追加
        traffic[ip_pair].append(packet)
        
    return traffic