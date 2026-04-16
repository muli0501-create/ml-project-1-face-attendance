"""
ONNX 模型导出脚本
=================
将训练好的 YOLO 模型导出为 ONNX 格式
"""

from ultralytics import YOLO
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.parent.absolute()


def export_yolo_to_onnx(model_path: str = None, imgsz: int = 640) -> str:
    """
    导出 YOLO 模型为 ONNX 格式

    Args:
        model_path: 模型路径，默认使用训练好的 best.pt
        imgsz: 输入图片大小

    Returns:
        str: 导出的 ONNX 文件路径
    """
    root = get_project_root()

    if model_path is None:
        model_path = root / "runs" / "detect" / "runs" / "face" / "exp12" / "weights" / "best.pt"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        print(f"错误：找不到模型文件 {model_path}")
        print("请检查模型路径是否正确")
        return None

    print(f"加载模型: {model_path}")
    model = YOLO(str(model_path))

    print(f"正在导出为 ONNX 格式 (imgsz={imgsz})...")
    export_path = model.export(format="onnx", imgsz=imgsz, opset=12)

    print(f"导出成功: {export_path}")
    return export_path


if __name__ == "__main__":
    print("=" * 50)
    print("YOLO 模型导出为 ONNX")
    print("=" * 50)

    root = get_project_root()
    default_model = root / "runs" / "detect" / "runs" / "face" / "exp12" / "weights" / "best.pt"

    if default_model.exists():
        print(f"使用默认模型: {default_model}")
        export_yolo_to_onnx()
    else:
        print(f"默认模型不存在: {default_model}")
        print("请手动指定模型路径")
