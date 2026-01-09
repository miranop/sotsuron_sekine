import os
import sys
from make_image.Recurrence import preprocess_series, sliding_windows
from make_image.Recurrence import generate_rp, save_rp_image
from make_image.utils import load_pcap
from make_image.utils.flow_util import group_by_pair
from make_image.utils.timeseries import series_per_second


from label import get_label, sanitize_label

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..")))


def process_pcap(filepath, is_train, tz_offset_hours=0.0):
    """
    pcap → group → series → window → RP → 保存
    + 攻撃ラベル付けも統合
    """
    pcap_name = os.path.basename(filepath).split('.')[0]
    print(f"\n▶ Processing {pcap_name} ({'train' if is_train else 'test'})")

    # 1. pcap を読み込む
    packets = load_pcap(filepath)
    if not packets:
        print("  ⚠ No packets.")
        return

    # 2. IP ペアごとにまとめる
    pairs = group_by_pair(packets)
    print(f"  IP pairs: {len(pairs)}")

    for pair_key, pkt_list in pairs.items():

        # 3. 時系列化（1秒ごと）
        series, secs = series_per_second(pkt_list)
        if len(series) < 256:
            continue

        # 4. 前処理（log1p + 正規化）
        series = preprocess_series(series)

        # 5. 小窓化（256 window, stride 64）
        windows = sliding_windows(series, window_size=256, stride=64)

        # 🔥 重要: 小窓ごとにラベルを付ける
        #     → 各ウィンドウの「最後の時刻」を使う
        #     → secs は series と 1:1 対応している時間（UNIX秒）
        for i, window in enumerate(windows):

            # ウィンドウの終端インデックス
            end_idx = min(i * 64 + 255, len(secs)-1)
            timestamp = secs[end_idx]

            # 攻撃ラベルを取得
            label_value = get_label(timestamp, filepath, tz_offset_hours)

            # ----------------------------
            #    RP 画像生成
            # ----------------------------
            img = generate_rp(window, image_dim=256)

            split = "train" if is_train else "test"

            # 連番取得

            # 保存
            save_rp_image(
                img,
                label_value,
                is_train,
                pcap_name=pcap_name,
            )

    print(f"  ✓ Done: {pcap_name}")
