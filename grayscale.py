from scapy.all import rdpcap, IP
import numpy as np
from PIL import Image
import os


def read_packet(filepath,count=1000):#パケットを
    ip_packets = []#分けたIPアドレスの格納用
    packets = rdpcap(filename=filepath,count=count)
    
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

def packets_to_images(packet_list, image_dim=64):
    """
    パケットのリストを64x64のグレースケール画像のリストに変換します。
    この時点ではまだ表示はせず、数値データのリストを返します。
    """
    images = []
    # パケットリストをimage_dim個ずつのチャンクに分割
    for i in range(0, len(packet_list), image_dim):
        chunk = packet_list[i:i + image_dim]
        
        # 64x64の、中身がすべて0の配列を作成 (画像のキャンバス)
        image_data = np.zeros((image_dim, image_dim), dtype=np.uint8)
        
        # チャンク内の各パケットを画像の1行に変換
        for j, packet in enumerate(chunk):
            # IPレイヤ以降をバイト列として取得
            packet_bytes = bytes(packet[IP])
            
            # 先頭から最大64バイトを取得
            header_bytes = packet_bytes[:image_dim]
            
            # 64バイトに満たない場合は255（白）で埋める
            padded_bytes = header_bytes.ljust(image_dim, b'\xff')
            
            # バイト列を数値(0-255)の配列に変換し、画像のj行目に設定
            image_data[j] = np.frombuffer(padded_bytes, dtype=np.uint8)
            
        # 最後のチャンクが64パケットに満たない場合、残りの行は白で埋める
        if len(chunk) < image_dim:
            image_data[len(chunk):] = 255 
        
        images.append(image_data)
        
    return images

def main():
    """メイン処理"""
    pcap_file = './Monday-WorkingHours.pcap'
    output_dir = "traffic_images"

    # 1. pcapからIPパケットを読み込む
    ip_packets = read_packet(pcap_file)
    if not ip_packets:
        return

    # 2. IPペアでパケットをグループ化
    traffic = group_packets(ip_packets)
    
    # 3. トラフィック量でソートし、上位5件を取得
    sorted_traffic = sorted(traffic.items(), key=lambda item: len(item[1]), reverse=True)
    top_pairs = sorted_traffic[:5]

    # 4. 出力用フォルダを作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: '{output_dir}'")

    # 5. 上位のIPペアのトラフィックを画像に変換して保存
    print("\n--- Converting traffic to images and saving ---")
    for ip_pair, packets in top_pairs:
        # 64パケット（=画像1枚分）以上の通信のみを対象
        if len(packets) >= 64:
            image_list = packets_to_images(packets)
            
            # ファイル名に使えない文字を置換
            ip1_sanitized = ip_pair[0].replace('.', '_')
            ip2_sanitized = ip_pair[1].replace('.', '_')
            base_filename = f"{ip1_sanitized}_{ip2_sanitized}"

            for idx, image_data in enumerate(image_list):
                # NumPy配列をPillowのImageオブジェクトに変換
                img = Image.fromarray(image_data, 'L') # 'L'はグレースケールモード
                
                # ファイルパスを構築して保存
                filename = os.path.join(output_dir, f"{base_filename}_{idx}.png")
                img.save(filename)
            
            print(f"Saved {len(image_list)} images for {ip_pair[0]} <-> {ip_pair[1]}")
        else:
            print(f"Skipping {ip_pair[0]} <-> {ip_pair[1]} (less than 64 packets)")
    print("---------------------------------------------")
if __name__ == '__main__':
    main()

    
