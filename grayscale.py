import numpy as np
import pandas as pd
import PIL.Image
from pathlib import Path


# --- CSV読み込み ---
def read_csv_to_dataframe(file_path):
    df = pd.read_csv(file_path, header=0)
    print(f"CSVファイル '{file_path}' を読み込みました。")
    df.columns = df.columns.str.strip()
    return df

print(read_csv_to_dataframe("./Monday-WorkingHours.pcap_ISCX.csv"))

# --- BENIGNフィルタリング & クリーニング ---
def filter_benign(df: pd.DataFrame) -> pd.DataFrame:
    lab = df["Label"].astype(str).str.strip().str.lower()
    out = df[lab == "benign"].copy()    
    print("BENIGN rows:", len(out))
    return out

#--- 必要列だけ取り出し、inf→NaN→drop ---
def get_clean_block(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return (
        df[cols]                              # 必要列だけ取り出す
        .replace([np.inf, -np.inf], np.nan)   # ±inf を NaN 扱いに
        .dropna()                             # NaN を含む行を全部落とす
    )


P_COLS = [
    "Flow Duration","Flow IAT Mean","Flow IAT Std","Fwd IAT Mean",
    "Flow Bytes/s","Flow Packets/s","Total Length of Fwd Packets","Total Length of Bwd Packets",
    "Fwd Packet Length Mean","Bwd Packet Length Mean","Fwd Packet Length Std","Bwd Packet Length Std",
    "SYN Flag Count","RST Flag Count","Fwd Header Length","Bwd Header Length"
]


# --- 1) BENIGNで分位点クリップ境界＆Min/Maxを学習（列まとめて） ---
def fit_minmax_params(benign_block: pd.DataFrame, q_low=0.01, q_high=0.99) -> dict:
    # 列ごとの分位点
    lo = benign_block.quantile(q_low) #1%
    hi = benign_block.quantile(q_high) #99%
    # クリップ後の分布で min/max（= スケーリング境界）
    clipped = benign_block.clip(lower=lo, upper=hi, axis=1)
    vmin = clipped.min()
    vmax = clipped.max()
    return {"lo": lo, "hi": hi, "vmin": vmin, "vmax": vmax}

# --- 2) 学習済みパラメータで、全データを 0..255 の uint8 にベクトル化変換 ---
def transform_to_uint8(block: pd.DataFrame, params: dict) -> pd.DataFrame:
    lo   = pd.Series(params["lo"])
    hi   = pd.Series(params["hi"])
    vmin = pd.Series(params["vmin"])
    vmax = pd.Series(params["vmax"])

    # 列順を固定（安全策）
    cols = vmin.index.tolist()
    b = block[cols].copy()

    # まず分位点でクリップ（BENIGN由来の境界）
    b = b.clip(lower=lo, upper=hi, axis=1)

    # Min–Max (0..1)
    denom = (vmax - vmin).replace(0, np.nan)
    scaled01 = (b - vmin) / denom
    scaled01 = scaled01.fillna(0.0).clip(0.0, 1.0)

    # 0..255 → uint8（完全ベクトル化）
    scaled255 = (scaled01 * 255.0).round().astype(np.uint8)
    # ざっくり飽和率（張り付き具合の監視）
    sat0 = (scaled255 == 0).mean().mean()
    sat1 = (scaled255 == 255).mean().mean()
    print(f"[transform_to_uint8] saturation 0/255 -> {sat0:.3f} / {sat1:.3f}")
    return scaled255

# --- 3) 1行 = 16値 を 4x4 グレースケール画像へ ---
def row_to_img4x4_uint8(row: pd.Series) -> PIL.Image.Image:
    arr = row.values.reshape(4, 4)
    return PIL.Image.fromarray(arr, mode="L")

# --- 4) 画像を書き出し ---
def export_images_uint8_matrix(df_uint8: pd.DataFrame, out_dir: str, limit: int = 20, prefix: str = "row"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    it = df_uint8.head(limit).iterrows() if limit is not None else df_uint8.iterrows()
    n = 0
    for i, (_, row) in enumerate(it):
        im = row_to_img4x4_uint8(row)
        im.save(out / f"{prefix}_{i:05d}.png")
        n += 1
    print(f"[export_images] saved {n} images -> {out.resolve()}")

# --- 5) gray_image 本体：fit(=BENIGN) → transform(=全体) → 保存 ---
def gray_image(df, out_dir="./out_gray", limit=20, q_low=0.01, q_high=0.99, save_params_path=None):
    # 1) BENIGN抽出 → クリーニング（inf→NaN→drop）
    df_b = filter_benign(df)
    benign_block = get_clean_block(df_b, P_COLS)
    print("[gray_image] BENIGN clean shape:", benign_block.shape)

    # 2) BENIGNで境界を学習
    params = fit_minmax_params(benign_block, q_low=q_low, q_high=q_high)
    print(f"[gray_image] fitted with q=({q_low}, {q_high})")

    # 3) 全データを同じ境界で 0..255 の uint8 に
    full_block = get_clean_block(df, P_COLS)  # 攻撃混在でもOK
    scaled_uint8 = transform_to_uint8(full_block, params)

    # 4) 先頭limit件だけ 4x4 画像にして書き出し（まずは目視チェック）
    export_images_uint8_matrix(scaled_uint8, out_dir=out_dir, limit=limit, prefix="row")

    # （任意）パラメータ保存したいとき
    if save_params_path:
        # pandas Series を辞書化して保存
        dumpable = {k: v.to_dict() for k, v in params.items()}
        import json
        with open(save_params_path, "w", encoding="utf-8") as f:
            json.dump(dumpable, f, ensure_ascii=False, indent=2)
        print(f"[gray_image] params saved -> {save_params_path}")

    print("[gray_image] done.")

if __name__ == "__main__":
    gray_image(read_csv_to_dataframe("./Monday-WorkingHours.pcap_ISCX.csv"))
