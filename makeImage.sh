#!/bin/bash

cd make_image

echo "========================================="
echo "全画像生成スクリプトを実行します"
echo "========================================="

echo -e "\n[1/5] グレースケール画像生成"
python grayscale.py

echo -e "\n[2/5] RGB画像生成"
python rgbscale.py

echo -e "\n[3/5] MTF画像生成"
python MTF.py

echo -e "\n[4/5] GAF画像生成"
python GAF.py

echo -e "\n[5/5] Recurrence Plot画像生成"
python Recurrence.py

echo -e "\n========================================="
echo "全画像生成完了！"
echo "========================================="