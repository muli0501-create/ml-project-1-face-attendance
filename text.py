import face_recognition
import numpy as np
from PIL import Image

# ========== 只改这一行！换成你自己的图片路径 ==========
img_path = "face_db/叶佳禾/01.jpeg"  # 替换成你实际的图片路径
# =====================================================

print("===== 开始检测face_recognition库 =====")
# 测试1：用库自带方法读取图片（核心检测）
try:
    img1 = face_recognition.load_image_file(img_path)
    print(f"✅ 测试1通过：load_image_file读取成功 | 格式：{img1.dtype} | 尺寸：{img1.shape}")
except Exception as e:
    print(f"❌ 测试1失败：load_image_file读取报错 → {e}（库/图片格式有问题）")

# 测试2：强制转标准8bit RGB后，库能否正常提取特征（兜底检测）
try:
    img2 = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    encs = face_recognition.face_encodings(img2)
    print(f"✅ 测试2通过：强制转格式后提取特征成功 | 提取到{len(encs)}个人脸特征")
except Exception as e:
    print(f"❌ 测试2失败：强制转格式后仍报错 → {e}（库本身环境兼容有问题）")

print("===== 检测结束 =====")