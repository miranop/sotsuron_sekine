from anomalib.data.utils import read_image
from anomalib.models import Padim

# テスト用のダミー画像生成
import numpy as np
from PIL import Image
import os

# ダミー画像保存
os.makedirs("test", exist_ok=True)
dummy_image = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))
dummy_image.save("test/test_img.jpg")

# Anomalib の read_image 関数とモデル動作確認
image = read_image("test/test_img.jpg")
model = Padim()
print("モデルと画像の読み込みに成功しました。")
