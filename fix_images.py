import cv2
import os
from pathlib import Path

# 人脸库根目录
FACE_DB_ROOT = "face_db"

# 遍历所有子文件夹（每个人的姓名文件夹）
for person_dir in Path(FACE_DB_ROOT).iterdir():
    if not person_dir.is_dir():
        continue
    # 遍历该人的所有图片
    for img_path in person_dir.glob("*.jpg"):
        try:
            # 读取图片（cv2.imread默认是BGR，我们转成RGB）
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"❌ 无法读取: {img_path}")
                continue
            # 转换为8bit RGB（cv2.imread本身就是8bit，这里做格式标准化）
            # 保存为标准JPG，覆盖原文件（也可以另存为新文件）
            cv2.imwrite(str(img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print(f"✅ 修复成功: {img_path}")
        except Exception as e:
            print(f"❌ 修复失败: {img_path}, 错误: {e}")

print("\n✅ 所有图片批量修复完成！")