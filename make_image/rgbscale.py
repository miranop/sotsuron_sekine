from scapy.all import rdpcap, IP
import numpy as np
from PIL import Image
import os

import same as same

# --- ステップ3' RGB画像変換用の新関数 ---
def packets_to_rgb_images(packet_list, image_dim=64):
    """パケットのリストを64x64のRGB画像のリストに変換する"""
    images = []
    packets_per_image = image_dim * 3  # 1枚の画像に必要なパケット数 (64*3=192)

    # 192パケットずつの塊（チャンク）に分割して処理
    for i in range(0, len(packet_list), packets_per_image):
        chunk = packet_list[i:i + packets_per_image]
        
        # 64x64x3 のRGB画像データ（numpy配列）を初期化
        image_data = np.zeros((image_dim, image_dim, 3), dtype=np.uint8)
        
        # チャンク内のパケットを画像の各行のRGBに変換
        for j in range(image_dim): # 0から63までループ (行番号)
            
            # 1行分のRGBデータを格納する配列
            r_row, g_row, b_row = np.zeros(image_dim), np.zeros(image_dim), np.zeros(image_dim)

            # --- Rチャンネルの処理 ---
            r_packet_index = j * 3
            if r_packet_index < len(chunk):
                packet_r = chunk[r_packet_index]
                header_bytes_r = bytes(packet_r[IP])
                padded_bytes_r = header_bytes_r.ljust(image_dim, b'\x00') # 足りない分は黒で埋める
                truncated_bytes_r = padded_bytes_r[:image_dim]
                r_row = np.frombuffer(truncated_bytes_r, dtype=np.uint8)

            # --- Gチャンネルの処理 ---
            g_packet_index = j * 3 + 1
            if g_packet_index < len(chunk):
                packet_g = chunk[g_packet_index]
                header_bytes_g = bytes(packet_g[IP])
                padded_bytes_g = header_bytes_g.ljust(image_dim, b'\x00')
                truncated_bytes_g = padded_bytes_g[:image_dim]
                g_row = np.frombuffer(truncated_bytes_g, dtype=np.uint8)

            # --- Bチャンネルの処理 ---
            b_packet_index = j * 3 + 2
            if b_packet_index < len(chunk):
                packet_b = chunk[b_packet_index]
                header_bytes_b = bytes(packet_b[IP])
                padded_bytes_b = header_bytes_b.ljust(image_dim, b'\x00')
                truncated_bytes_b = padded_bytes_b[:image_dim]
                b_row = np.frombuffer(truncated_bytes_b, dtype=np.uint8)

            # numpy配列の対応するチャンネルにデータを格納
            image_data[j, :, 0] = r_row  # j行目のRチャンネル
            image_data[j, :, 1] = g_row  # j行目のGチャンネル
            image_data[j, :, 2] = b_row  # j行目のBチャンネル

        images.append(image_data)
        
    return images

# --- メイン処理 ---
def main():
    """メイン処理"""
    pcap_file = '../Monday-WorkingHours.pcap'
    output_dir = "traffic_images_rgb" # 保存先フォルダ名を変更

    # 1, 2. パケット読み込みとグループ化 (変更なし)
    ip_packets = same.read_packet(pcap_file)
    if not ip_packets:
        return
    traffic = same.group_packets(ip_packets)
    
    # 3. トラフィック量でソート
    sorted_traffic = sorted(traffic.items(), key=lambda item: len(item[1]), reverse=True)
    
    # 4. 出力用フォルダを作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: '{output_dir}'")

    # 5. トラフィックをRGB画像に変換して保存
    print("\n--- Converting traffic to RGB images and saving ---")
    for ip_pair, packets in sorted_traffic:
        # 192パケット（=RGB画像1枚分）以上の通信のみを対象
        if len(packets) >= 192:
            image_list = packets_to_rgb_images(packets)
            
            ip1_sanitized = ip_pair[0].replace('.', '_')
            ip2_sanitized = ip_pair[1].replace('.', '_')
            base_filename = f"{ip1_sanitized}_{ip2_sanitized}"

            for idx, image_data in enumerate(image_list):
                # NumPy配列をPillowのImageオブジェクトに変換 (モードを'RGB'に変更)
                img = Image.fromarray(image_data, 'RGB') 
                
                filename = os.path.join(output_dir, f"{base_filename}_{idx}.png")
                img.save(filename)
            
            print(f"Saved {len(image_list)} RGB images for {ip_pair[0]} <-> {ip_pair[1]}")
    print("---------------------------------------------")

if __name__ == "__main__":
    main()
