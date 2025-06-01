from scapy.all import PcapReader
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from PIL import Image

def get_protocol_type(packet):
    """パケットのプロトコルタイプを判定"""
    if packet.haslayer('TCP'):
        return "TCP"
    elif packet.haslayer('UDP'):
        return "UDP"
    elif packet.haslayer('ICMP'):
        return "ICMP"
    elif packet.haslayer('ARP'):
        return "ARP"
    elif packet.haslayer('IPv6'):
        return "IPv6"
    else:
        return "Other"

def analyze_pcap_distribution(pcap_file, max_analyze_packets=10000):
    """PCAPファイルの分布を分析（指定した数まで）"""
    protocol_counts = []
    total_packets = 0
    packet_sizes = []
    
    print(f"PCAPファイルを分析中（最大{max_analyze_packets}パケット）...")
    with PcapReader(pcap_file) as pcap_reader:
        for packet in pcap_reader:
            total_packets += 1
            protocol = get_protocol_type(packet)
            protocol_counts[protocol] += 1
            packet_sizes.append(len(packet))
            
            # 進捗表示（1000パケットごと）
            if total_packets % 1000 == 0:
                print(f"  分析済み: {total_packets} パケット")
            
            # 指定した数に達したら停止
            if total_packets >= max_analyze_packets:
                break
    
    print(f"\n=== PCAP分析結果 ===")
    print(f"分析パケット数: {total_packets}")
    print(f"平均パケットサイズ: {np.mean(packet_sizes):.2f} bytes")
    print(f"パケットサイズ範囲: {min(packet_sizes)} - {max(packet_sizes)} bytes")
    print("\nプロトコル別分布:")
    for protocol, count in sorted(protocol_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_packets) * 100
        print(f"  {protocol}: {count} ({percentage:.2f}%)")
    
    return protocol_counts, total_packets

def stratified_sampling(pcap_file, total_samples=1000, min_samples_per_protocol=5):
    """ストラティファイドサンプリング実行"""
    # 第1パス：分布分析
    protocol_counts, total_packets = analyze_pcap_distribution(pcap_file)
    
    # 各プロトコルのサンプル数を決定（比例配分 + 最小保証）
    protocol_samples = {}
    remaining_samples = total_samples
    
    # まず各プロトコルに最小数を保証
    for protocol in protocol_counts.keys():
        min_samples = min(min_samples_per_protocol, protocol_counts[protocol])
        protocol_samples[protocol] = min_samples
        remaining_samples -= min_samples
    
    # 残りを比例配分
    if remaining_samples > 0:
        for protocol, count in protocol_counts.items():
            ratio = count / total_packets
            additional_samples = int(remaining_samples * ratio)
            protocol_samples[protocol] += additional_samples
            
    # 数の調整
    current_total = sum(protocol_samples.values())
    total_collected = sum(protocol_samples.values())
    if current_total < total_samples:
            shortage = total_samples - current_total
            # 最も多いプロトコルに不足分を追加
            largest_protocol = max(protocol_counts.items(), key=lambda x: x[1])[0]
            protocol_samples[largest_protocol] += shortage
            print(f"  端数調整: {largest_protocol}に{shortage}サンプル追加")
    
    print(f"\n=== サンプリング計画 ===")
    print(f"目標サンプル数: {total_samples}")
    for protocol, samples in protocol_samples.items():
        print(f"  {protocol}: {samples} samples")
    
    # 第2パス：実際のサンプリング
    sampled_packets = []
    protocol_collected = {p: 0 for p in protocol_samples.keys()}
    
    print(f"\nサンプリング実行中...")
    with PcapReader(pcap_file) as pcap_reader:
        for i, packet in enumerate(pcap_reader):
            protocol = get_protocol_type(packet)
            
            if (protocol in protocol_collected and 
                protocol_collected[protocol] < protocol_samples[protocol]):
                sampled_packets.append(packet)
                protocol_collected[protocol] += 1
                
            # 進捗表示
            if i % 10000 == 0 and i > 0:
                total_collected = sum(protocol_collected.values())
                print(f"  処理済み: {i} パケット, 収集済み: {total_collected}/{total_samples}")
                
            # 全て収集完了なら終了
            if sum(protocol_collected.values()) >= total_samples:
                break
    
    final_count = sum(protocol_collected.values())
    print(f"\nサンプリング完了: {final_count} パケット収集")
    for protocol, count in protocol_collected.items():
        if count > 0:
            print(f"  {protocol}: {count}")
    
    return sampled_packets

def packet_to_bytes_image(packet, size=(32, 32)):
    """パケットをバイト配列から画像に変換"""
    packet_bytes = bytes(packet)
    target_size = size[0] * size[1]
    
    # 固定サイズにパディング/切り詰め
    if len(packet_bytes) < target_size:
        packet_bytes += b'\x00' * (target_size - len(packet_bytes))
    else:
        packet_bytes = packet_bytes[:target_size]
    
    # numpy配列に変換して画像化
    img_array = np.frombuffer(packet_bytes, dtype=np.uint8)
    return img_array.reshape(size)

def create_packet_images(packets, image_size=(32, 32)):
    """パケットリストを画像データセットに変換"""
    print(f"\n画像データセット作成中...")
    images = []
    labels = []
    
    for i, packet in enumerate(packets):
        # パケットを画像に変換
        img = packet_to_bytes_image(packet, image_size)
        images.append(img)
        
        # ラベル（プロトコルタイプ）を記録
        protocol = get_protocol_type(packet)
        labels.append(protocol)
        
        if (i + 1) % 100 == 0:
            print(f"  変換済み: {i + 1}/{len(packets)} パケット")
    
    images = np.array(images)
    return images, labels

def visualize_sample_images(images, labels, n_samples=9):
    """サンプル画像を可視化"""
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle('Sample Packet Images', fontsize=16)
    
    # ランダムにサンプルを選択
    indices = random.sample(range(len(images)), min(n_samples, len(images)))
    
    for i, idx in enumerate(indices):
        row = i // 3
        col = i % 3
        
        axes[row, col].imshow(images[idx], cmap='gray')
        axes[row, col].set_title(f'{labels[idx]}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_dataset(images, labels, prefix='packet_dataset'):
    """データセットを保存"""
    np.save(f'{prefix}_images.npy', images)
    np.save(f'{prefix}_labels.npy', labels)
    
    print(f"\nデータセット保存完了:")
    print(f"  画像: {prefix}_images.npy ({images.shape})")
    print(f"  ラベル: {prefix}_labels.npy ({len(labels)} labels)")

# 使用例
if __name__ == "__main__":
    # PCAPファイルのパス
    pcap_file = "Monday-WorkingHours.pcap"
    
    # ストラティファイドサンプリング実行
    sampled_packets = stratified_sampling(pcap_file, total_samples=997)
    
    # パケットを画像に変換
    images, labels = create_packet_images(sampled_packets, image_size=(32, 32))
    
    # サンプル画像を表示
    visualize_sample_images(images, labels)
    
    # データセットを保存
    save_dataset(images, labels, 'normal_traffic_dataset')
    
    print("\n処理完了!")