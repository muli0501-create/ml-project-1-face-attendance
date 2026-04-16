"""
人脸注册模块 - 将未知人脸即时加入人脸库
"""

import os
import pickle
import shutil
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent.absolute()


def load_face_db():
    """加载人脸特征库"""
    root = get_project_root()
    encodings_path = root / "face_db" / "encodings.pkl"
    if encodings_path.exists():
        try:
            with open(encodings_path, "rb") as f:
                return pickle.load(f)
        except:
            return {}
    return {}


def save_face_db(face_db):
    """保存人脸特征库"""
    root = get_project_root()
    encodings_path = root / "face_db" / "encodings.pkl"
    with open(encodings_path, "wb") as f:
        pickle.dump(face_db, f)


def register_new_face(name: str, image_paths: list, copy_images: bool = True) -> dict:
    """注册新用户人脸"""
    root = get_project_root()
    face_db_dir = root / "face_db"

    if not name or not name.strip():
        return {"success": False, "message": "姓名不能为空", "encodings_count": 0}

    name = name.strip()
    person_dir = face_db_dir / name

    # 验证图片
    valid_paths = []
    for p in image_paths:
        path = Path(p)
        if path.exists() and path.is_file():
            ext = path.suffix.lower()
            if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
                valid_paths.append(path)

    if len(valid_paths) == 0:
        return {"success": False, "message": "没有找到有效的图片文件", "encodings_count": 0}

    # 创建用户目录
    if copy_images:
        person_dir.mkdir(parents=True, exist_ok=True)

    # 导入 face_recognition
    import face_recognition

    encodings = []
    failed_files = []
    success_count = 0

    for img_path in valid_paths:
        try:
            # 使用 face_recognition 加载图片（支持中文路径）
            image = face_recognition.load_image_file(str(img_path))
            encs = face_recognition.face_encodings(image)

            if encs:
                encodings.append(encs[0])
                success_count += 1
            else:
                failed_files.append(f"{img_path.name} (未检测到人脸)")

            # 复制图片到人脸库
            if copy_images:
                dest_path = person_dir / img_path.name
                if dest_path.exists():
                    base = img_path.stem
                    ext = img_path.suffix
                    counter = 1
                    while dest_path.exists():
                        dest_path = person_dir / f"{base}_{counter}{ext}"
                        counter += 1
                shutil.copy2(img_path, dest_path)

        except Exception as e:
            failed_files.append(f"{img_path.name} ({str(e)})")

    if len(encodings) == 0:
        msg = "未能提取到人脸编码，请确保照片中包含清晰的人脸"
        return {"success": False, "message": msg, "encodings_count": 0}

    # 更新人脸库
    face_db = load_face_db()
    face_db[name] = encodings
    save_face_db(face_db)

    msg = f"成功注册 {name}，共 {success_count} 张照片入库"
    if failed_files:
        msg += f"\n失败: {', '.join(failed_files[:3])}"

    return {
        "success": True,
        "message": msg,
        "encodings_count": len(encodings),
    }


def register_from_uploaded_images(name: str, uploaded_files: list) -> dict:
    """从 Streamlit 上传的文件注册人脸"""
    import tempfile

    if not name or not name.strip():
        return {"success": False, "message": "姓名不能为空", "encodings_count": 0}

    name = name.strip()

    # 使用临时目录保存上传的文件
    temp_dir = Path(tempfile.mkdtemp())
    temp_paths = []

    try:
        for i, uploaded_file in enumerate(uploaded_files):
            # 保留原文件扩展名
            ext = Path(uploaded_file.name).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".bmp"}:
                ext = ".jpg"
            temp_path = temp_dir / f"photo_{i+1}{ext}"

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            temp_paths.append(temp_path)

        result = register_new_face(name, temp_paths, copy_images=True)
        return result

    finally:
        # 清理临时文件
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def rebuild_face_db() -> dict:
    """重新构建人脸特征库（从 face_db 目录读取所有图片）"""
    root = get_project_root()
    face_db_dir = root / "face_db"

    if not face_db_dir.exists():
        return {"success": False, "message": "face_db 目录不存在", "count": 0}

    import face_recognition

    face_db = {}
    total_count = 0

    # 遍历所有人脸目录
    for person_dir in face_db_dir.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        encodings = []

        # 遍历该目录下的所有图片
        for img_file in person_dir.iterdir():
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            try:
                image = face_recognition.load_image_file(str(img_file))
                encs = face_recognition.face_encodings(image)
                if encs:
                    encodings.append(encs[0])
            except Exception as e:
                print(f"处理 {person_name}/{img_file.name} 失败: {e}")

        if encodings:
            face_db[person_name] = encodings
            total_count += len(encodings)
            print(f"✓ {person_name}: {len(encodings)} 张照片")

    if face_db:
        save_face_db(face_db)

    return {
        "success": True,
        "message": f"重建完成，共 {len(face_db)} 人，{total_count} 个编码",
        "count": total_count,
        "people": len(face_db)
    }


if __name__ == "__main__":
    # 测试重建人脸库
    print("重新构建人脸特征库...")
    result = rebuild_face_db()
    print(result)
