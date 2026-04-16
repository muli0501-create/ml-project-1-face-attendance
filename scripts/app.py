"""
Streamlit 人脸签到系统 Web 界面
===============================
支持：图片上传签到 | 未知人脸注册 | 签到可视化 | ONNX导出

运行：streamlit run scripts/app.py
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pickle
import tempfile
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime


# ============ 修复 face_recognition 中文路径问题 ============
def _fix_face_recognition_chinese_path():
    """
    face_recognition 在中文路径下会出错，因为 dlib 模型路径包含中文。
    解决方案：复制模型文件到临时目录，并修改 face_recognition_models 的路径函数。
    """
    try:
        import site
        site_packages = [p for p in sys.path if 'site-packages' in p][0]
        fr_models_pkg_path = Path(site_packages) / "face_recognition_models"
        fr_models_files_path = fr_models_pkg_path / "models"

        if not fr_models_files_path.exists():
            print("未找到 face_recognition_models")
            return

        # 创建临时目录（无中文）
        temp_dir = Path(tempfile.gettempdir()) / "face_rec_models"
        temp_dir.mkdir(exist_ok=True)

        # 复制所有模型文件到临时目录
        for model_file in fr_models_files_path.glob("*.dat"):
            temp_file = temp_dir / model_file.name
            if not temp_file.exists() or temp_file.stat().st_size != model_file.stat().st_size:
                shutil.copy2(model_file, temp_file)
            print(f"已复制模型: {model_file.name} -> {temp_file}")

        # 修改 face_recognition_models/__init__.py
        init_path = fr_models_pkg_path / "__init__.py"
        with open(init_path, 'r', encoding='utf-8') as f:
            init_code = f.read()

        if 'TEMP_MODEL_PATH' in init_code:
            return  # 已经修复过

        # 替换为返回临时目录路径
        temp_dir_str = str(temp_dir).replace('\\', '\\\\')
        new_init_code = init_code.replace(
            'from pkg_resources import resource_filename',
            f'_TEMP_MODEL_PATH = r"{temp_dir_str}"\nfrom pkg_resources import resource_filename'
        ).replace(
            'return resource_filename(__name__, "models/shape_predictor_68_face_landmarks.dat")',
            f'return _TEMP_MODEL_PATH + "\\\\\\\\shape_predictor_68_face_landmarks.dat"'
        ).replace(
            'return resource_filename(__name__, "models/shape_predictor_5_face_landmarks.dat")',
            f'return _TEMP_MODEL_PATH + "\\\\\\\\shape_predictor_5_face_landmarks.dat"'
        ).replace(
            'return resource_filename(__name__, "models/dlib_face_recognition_resnet_model_v1.dat")',
            f'return _TEMP_MODEL_PATH + "\\\\\\\\dlib_face_recognition_resnet_model_v1.dat"'
        ).replace(
            'return resource_filename(__name__, "models/mmod_human_face_detector.dat")',
            f'return _TEMP_MODEL_PATH + "\\\\\\\\mmod_human_face_detector.dat"'
        )

        # 备份并写入
        backup_path = init_path.with_suffix('.py.bak')
        if not backup_path.exists():
            shutil.copy2(init_path, backup_path)

        with open(init_path, 'w', encoding='utf-8') as f:
            f.write(new_init_code)

        print(f"face_recognition 模型路径已修复为: {temp_dir}")
    except Exception as e:
        print(f"修复 face_recognition 路径失败: {e}")


# 在任何 face_recognition 导入之前先修复
_fix_face_recognition_chinese_path()


# 页面配置
st.set_page_config(
    page_title="人脸签到系统",
    page_icon="📸",
    layout="wide"
)


# ============ 路径和模型加载 ============
@st.cache_resource
def get_project_root():
    return Path(__file__).parent.parent.absolute()


@st.cache_resource
def load_models():
    """加载 YOLO 模型和人脸特征库"""
    root = get_project_root()

    # YOLO 模型
    yolo_path = root / "runs" / "detect" / "runs" / "face" / "exp12" / "weights" / "best.pt"
    if not yolo_path.exists():
        return None, None, None, "YOLO模型不存在"

    try:
        from ultralytics import YOLO
        yolo_model = YOLO(str(yolo_path))
    except Exception as e:
        return None, None, None, f"YOLO加载失败: {e}"

    # 人脸特征库
    encodings_path = root / "face_db" / "encodings.pkl"
    if not encodings_path.exists():
        return yolo_model, None, None, "人脸特征库不存在，请先运行 build_face_db.py"

    try:
        with open(encodings_path, "rb") as f:
            face_db = pickle.load(f)

        known_names, known_encodings = [], []
        for name, encs in face_db.items():
            for enc in encs:
                known_names.append(name)
                known_encodings.append(enc)
    except Exception as e:
        return yolo_model, None, None, f"人脸库加载失败: {e}"

    return yolo_model, known_names, known_encodings, None


# ============ 人脸检测与识别 ============
def detect_and_recognize(image, yolo_model, known_names, known_encodings,
                         confidence=0.5, tolerance=0.5):
    """对图片进行人脸检测和识别"""
    if yolo_model is None:
        return None, []

    # 确保是 BGR 格式
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    # YOLO 检测
    results = yolo_model(image, conf=confidence, verbose=False)
    boxes = results[0].boxes

    attendance = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # 裁剪人脸
        face_crop = image[max(0, y1):min(image.shape[0], y2),
                         max(0, x1):min(image.shape[1], x2)]

        name = "Unknown"
        if face_crop.size > 0:
            try:
                import face_recognition
                rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb_crop)
                if encs:
                    dists = face_recognition.face_distance(known_encodings, encs[0])
                    if len(dists) > 0:
                        best_idx = np.argmin(dists)
                        if dists[best_idx] < tolerance:
                            name = known_names[best_idx]
            except:
                pass

        attendance.append({
            "姓名": name,
            "置信度": f"{conf:.2f}",
            "位置": f"({x1}, {y1}, {x2}, {y2})"
        })

        # 绘制边框
        color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 添加统计信息
    total_detected = len(boxes)
    checked_in = len([a for a in attendance if a["姓名"] != "Unknown"])
    stat_text = f"检测: {total_detected} 人 | 签到: {checked_in} 人"
    cv2.putText(image, stat_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return image, attendance


def load_image_from_upload(uploaded_file):
    """从 Streamlit 上传文件加载图片"""
    if uploaded_file is None:
        return None

    # 读取文件内容
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    return image


# ============ 签到记录加载 ============
@st.cache_data
def load_attendance_logs(log_dir):
    """加载签到记录"""
    root = get_project_root()
    if log_dir is None:
        log_dir = root / "attendance_log"

    all_records = []
    if not log_dir.exists():
        return pd.DataFrame()

    for csv_file in log_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, encoding="utf-8-sig")
            df["来源文件"] = csv_file.name
            all_records.append(df)
        except:
            pass

    if not all_records:
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)
    combined["签到时间_dt"] = pd.to_datetime(combined["签到时间"], errors="coerce")
    combined["日期"] = combined["签到时间_dt"].dt.date
    combined["小时"] = combined["签到时间_dt"].dt.hour
    return combined


# ============ 页面：签到系统 ============
def page_attendance():
    st.header("📸 人脸签到")

    # 选择签到模式
    mode = st.radio("选择签到模式", ["📁 上传图片签到", "📷 摄像头实时签到"], horizontal=True)
    st.divider()

    if mode == "📁 上传图片签到":
        page_attendance_upload()
    else:
        page_attendance_camera()


def page_attendance_upload():
    """图片上传签到"""
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("上传图片")
        uploaded_files = st.file_uploader(
            "选择图片文件",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.success(f"已上传 {len(uploaded_files)} 张图片")

    with col2:
        st.subheader("签到结果")
        result_area = st.empty()

    if uploaded_files:
        yolo_model, known_names, known_encodings, error = load_models()

        if error:
            st.error(error)
            return

        for uploaded_file in uploaded_files:
            st.divider()
            st.write(f"**处理: {uploaded_file.name}**")

            image = load_image_from_upload(uploaded_file)
            if image is None:
                st.error(f"无法读取图片: {uploaded_file.name}")
                continue

            # 人脸检测与识别
            result_img, attendance = detect_and_recognize(
                image.copy(), yolo_model, known_names, known_encodings
            )

            # 显示结果图片
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, channels="RGB", width=500)

            # 显示签到名单
            if attendance:
                df_result = pd.DataFrame(attendance)
                st.dataframe(df_result, use_container_width=True)

                # 统计
                known_count = len([a for a in attendance if a["姓名"] != "Unknown"])
                st.info(f"检测到 {len(attendance)} 人，已签到 {known_count} 人")
            else:
                st.warning("未检测到人脸")


def page_attendance_camera():
    """摄像头实时签到"""
    st.subheader("摄像头实时签到")

    # 初始化 session state
    if "attendance_record" not in st.session_state:
        st.session_state.attendance_record = {}
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False
    if "frame_placeholder" not in st.session_state:
        st.session_state.frame_placeholder = None

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("**实时画面**")
        frame_placeholder = st.image([], width=400)

        # 签到状态显示
        status_text = st.empty()

        # 控制按钮
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("▶️ 开始签到", type="primary"):
                st.session_state.camera_running = True
        with btn_col2:
            if st.button("⏹️ 结束签到"):
                st.session_state.camera_running = False

    with col2:
        st.write("**已签到人员**")
        record_placeholder = st.empty()

        if st.session_state.attendance_record:
            records = [{"姓名": name, "签到时间": time}
                      for name, time in st.session_state.attendance_record.items()]
            df = pd.DataFrame(records)
            record_placeholder.dataframe(df, use_container_width=True)
            st.success(f"共 {len(st.session_state.attendance_record)} 人签到")
        else:
            record_placeholder.info("暂无签到记录")

        # 导出按钮
        if st.session_state.attendance_record:
            if st.button("📥 导出签到记录"):
                export_attendance_from_camera()
                st.success("签到记录已导出到 attendance_log/ 目录")

    # 实时摄像头处理
    if st.session_state.camera_running:
        run_camera_attendance(frame_placeholder, status_text)


def run_camera_attendance(frame_placeholder, status_text):
    """运行摄像头实时签到"""
    import face_recognition

    yolo_model, known_names, known_encodings, error = load_models()

    if error:
        st.error(error)
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("无法打开摄像头，请检查摄像头是否连接")
        return

    status_text.info("正在检测人脸...")

    frame_count = 0
    skip_frames = 5  # 每隔N帧识别一次

    try:
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.error("无法读取摄像头画面")
                break

            frame_count += 1

            # YOLO 检测
            results = yolo_model(frame, conf=0.5, verbose=False)
            boxes = results[0].boxes

            # 每隔几帧识别一次
            if frame_count % (skip_frames + 1) == 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    face_crop = frame[max(0, y1):min(frame.shape[0], y2),
                                     max(0, x1):min(frame.shape[1], x2)]

                    name = "Unknown"
                    if face_crop.size > 0:
                        try:
                            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            encs = face_recognition.face_encodings(rgb_crop)
                            if encs:
                                dists = face_recognition.face_distance(known_encodings, encs[0])
                                if len(dists) > 0:
                                    best_idx = np.argmin(dists)
                                    if dists[best_idx] < 0.5:
                                        name = known_names[best_idx]
                        except:
                            pass

                    # 签到记录
                    if name != "Unknown" and name not in st.session_state.attendance_record:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.attendance_record[name] = now
                        st.success(f"✓ {name} 签到成功！")

                    # 绘制边框
                    color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, name, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 添加统计信息
            total = len(st.session_state.attendance_record)
            cv2.putText(frame, f"Checked In: {total}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 显示画面
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", width=400)

    finally:
        cap.release()


def export_attendance_from_camera():
    """导出摄像头签到记录"""
    root = get_project_root()
    output_dir = root / "attendance_log"
    output_dir.mkdir(exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"attendance_{date_str}.csv"
    filepath = output_dir / filename

    records = [{"序号": i+1, "姓名": name, "签到时间": time}
               for i, (name, time) in enumerate(st.session_state.attendance_record.items())]

    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False, encoding="utf-8-sig")

    return filepath


# ============ 页面：人脸注册 ============
def page_register():
    st.header("👤 未知人脸注册")

    sys.path.insert(0, str(Path(__file__).parent))
    import register_face as rf

    # 显示当前已注册人数
    face_db = rf.load_face_db()
    if face_db:
        with st.expander(f"当前已注册人员 ({len(face_db)} 人)"):
            for name, encs in face_db.items():
                st.write(f"  • {name}: {len(encs)} 张照片")

    # 选择录入方式
    reg_mode = st.radio("选择录入方式", ["📁 上传照片", "📷 摄像头拍照"], horizontal=True)

    name = st.text_input("姓名", placeholder="请输入姓名")

    if reg_mode == "📁 上传照片":
        page_register_upload(name, rf)
    else:
        page_register_camera(name, rf)


def page_register_upload(name, rf):
    """照片上传方式注册"""
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("上传照片")
        uploaded_photos = st.file_uploader(
            "选择照片 (3-5张)",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True
        )

        if uploaded_photos:
            st.write(f"已上传 {len(uploaded_photos)} 张照片")

        register_btn = st.button("注册", type="primary",
                                  disabled=not (name and uploaded_photos))

    with col2:
        st.subheader("注册说明")
        st.info("""
        **注册要求：**
        1. 上传 3-5 张清晰正面照片
        2. 照片中只有一个正脸
        3. 输入真实姓名
        4. 点击注册按钮完成
        """)
        st.warning("注意：注册后该用户将能被人脸识别系统识别")

    if register_btn:
        with st.spinner("正在注册..."):
            result = rf.register_from_uploaded_images(name, uploaded_photos)
        if result["success"]:
            st.success(result["message"])
            # 清除缓存以刷新人脸库
            st.cache_resource.clear()
            # 清空已上传的照片
            st.rerun()
        else:
            st.error(result["message"])


def page_register_camera(name, rf):
    """摄像头拍照方式注册"""
    st.subheader("摄像头拍照")

    # 初始化 session state
    if "captured_photos" not in st.session_state:
        st.session_state.captured_photos = []
    if "camera_key" not in st.session_state:
        st.session_state.camera_key = 0

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("**拍摄照片（需要 3-5 张）**")

        # 摄像头输入
        camera_image = st.camera_input(
            f"拍摄第 {len(st.session_state.captured_photos) + 1} 张",
            key=f"camera_{st.session_state.camera_key}"
        )

        if camera_image is not None:
            # 添加到已拍摄列表
            if len(st.session_state.captured_photos) < 10:  # 最多10张
                st.session_state.captured_photos.append(camera_image)
                st.success(f"已拍摄 {len(st.session_state.captured_photos)} 张")

        # 显示已拍摄的照片
        if st.session_state.captured_photos:
            st.write("**已拍摄照片：**")
            cols = st.columns(min(len(st.session_state.captured_photos), 5))
            for i, photo in enumerate(st.session_state.captured_photos):
                with cols[i % 5]:
                    st.image(photo, width=100)

        # 按钮行
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            clear_btn = st.button("🗑️ 清空照片", disabled=not st.session_state.captured_photos)
        with btn_col2:
            retake_btn = st.button("📷 重新拍摄")

        if clear_btn:
            st.session_state.captured_photos = []
            st.rerun()
        if retake_btn:
            st.session_state.captured_photos = []
            st.session_state.camera_key += 1
            st.rerun()

    with col2:
        st.subheader("拍照说明")
        st.info("""
        **拍照要求：**
        1. 正对摄像头，保持面部清晰
        2. 建议拍摄 3-5 张不同角度
        3. 光线充足，避免逆光
        4. 表情自然，避免夸张动作
        """)
        st.warning("注意：注册后该用户将能被人脸识别系统识别")

    # 注册按钮
    st.divider()
    register_btn = st.button("✅ 确认注册",
                              type="primary",
                              disabled=not (name and len(st.session_state.captured_photos) >= 1))

    if register_btn:
        with st.spinner("正在注册..."):
            result = rf.register_from_uploaded_images(name, st.session_state.captured_photos)

        if result["success"]:
            st.success(result["message"])
            # 清除缓存以刷新人脸库
            st.cache_resource.clear()
            st.session_state.captured_photos = []
            st.session_state.camera_key += 1
            st.rerun()
        else:
            st.error(result["message"])


# ============ 页面：数据可视化 ============
def page_visualization():
    st.header("📊 签到记录可视化")

    sys.path.insert(0, str(Path(__file__).parent))
    import visualize_attendance as va

    # 加载数据
    df = load_attendance_logs(None)

    if df.empty:
        st.warning("暂无签到记录")
        return

    st.success(f"共加载 {len(df)} 条签到记录")

    # 显示原始数据
    with st.expander("查看原始数据"):
        st.dataframe(df[["序号", "姓名", "签到时间"]].sort_values("签到时间", ascending=False))

    # 生成图表
    fig = va.plot_attendance_stats(df)

    # 显示图表
    st.plotly_chart(fig, use_container_width=True)

    # 详细统计
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        total_people = df["姓名"].nunique() if "姓名" in df.columns else 0
        st.metric("注册人数", total_people)

    with col2:
        total_checkins = len(df)
        st.metric("总签到次数", total_checkins)

    with col3:
        if "日期" in df.columns:
            days = df["日期"].nunique()
            st.metric("签到天数", days)


# ============ 页面：ONNX导出 ============
def page_export_onnx():
    st.header("📦 模型导出 ONNX")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("导出设置")

        # 显示当前模型信息
        root = get_project_root()
        model_path = root / "runs" / "detect" / "runs" / "face" / "exp12" / "weights" / "best.pt"

        if model_path.exists():
            st.success(f"✅ 找到训练好的模型")
            st.write(f"**模型路径:** `{model_path}`")
            import os
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            st.write(f"**模型大小:** {size_mb:.2f} MB")
        else:
            st.error(f"❌ 模型文件不存在: {model_path}")
            return

        imgsz = st.selectbox("输入图片大小", [320, 480, 640, 800], index=2)

        export_btn = st.button("导出 ONNX", type="primary")

    with col2:
        st.subheader("导出说明")
        st.info("""
        **ONNX 格式优势：**
        - 跨平台部署
        - 支持 ONNX Runtime 推理
        - 可转换为其他框架格式
        - 不依赖 PyTorch 环境
        """)

        st.code("""
        # ONNX 推理示例
        import onnxruntime as ort
        session = ort.InferenceSession("best.onnx")
        results = session.run(None, {"input": image})
        """)

    if export_btn:
        with st.spinner("正在导出..."):
            try:
                from ultralytics import YOLO
                model = YOLO(str(model_path))
                export_path = model.export(format="onnx", imgsz=imgsz, opset=12)

                st.success(f"✅ 导出成功！")
                st.write(f"**ONNX 文件:** `{export_path}`")

                import os
                if os.path.exists(export_path):
                    size_mb = os.path.getsize(export_path) / (1024 * 1024)
                    st.write(f"**文件大小:** {size_mb:.2f} MB")

                    # 提供下载链接
                    with open(export_path, "rb") as f:
                        st.download_button(
                            "下载 ONNX 模型",
                            f,
                            os.path.basename(export_path),
                            "application/octet-stream"
                        )
            except Exception as e:
                st.error(f"导出失败: {e}")


# ============ 主程序 ============
def main():
    st.title("📸 人脸签到系统")

    # 侧边栏导航 - 垂直排列
    st.sidebar.title("功能导航")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "",
        ["📸 图片签到", "👤 人脸注册", "📊 数据可视化", "📦 ONNX导出"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("基于 YOLOv11 + face_recognition")

    if page == "📸 图片签到":
        page_attendance()
    elif page == "👤 人脸注册":
        page_register()
    elif page == "📊 数据可视化":
        page_visualization()
    elif page == "📦 ONNX导出":
        page_export_onnx()


if __name__ == "__main__":
    main()
