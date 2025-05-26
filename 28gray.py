import numpy as np
import pandas as pd
import PIL.Image

def read_csv_to_dataframe(file_path):
    df = pd.read_csv(file_path, header=0,nrows=1)
    print(f"CSVファイル '{file_path}' を読み込みました。")
    return df

print(read_csv_to_dataframe("./Monday-WorkingHours.pcap_ISCX.csv"))

def gray_image(row ,label ,output = "sample.png",size=28):
    fe = row.drop(labels=[" Label"]).values.astype(np.float32)
    padded = np.pad(fe, (0, size*size - len(fe)), 'constant')
    norm = (padded - np.min(padded)) / (np.max(padded) - np.min(padded) + 1e-5) #正規化
    gray = (norm * 255).astype(np.uint8).reshape((size, size)) #28x28のグレースケール画像に変換
    img = PIL.Image.fromarray(np.uint8(gray), mode='L')  
    img.save(output)


if __name__ == "__main__":
    df = read_csv_to_dataframe("./Monday-WorkingHours.pcap_ISCX.csv")
    print(df.columns.tolist())

    row = df.iloc[0]
    label = row[" Label"] if "Label" in row else "unknown"
    gray_image(row, label, output="sample.png", size=28)
    print("グレースケール画像を 'sample.png' として保存しました。")

