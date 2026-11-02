import argparse
import time
from pathlib import Path
from ultralytics import YOLO

def evaluate(weights_path=None, split="test", imgsz=640, device=""):
    project_root = Path(__file__).resolve().parent.parent
    if weights_path is None:
        default_path = project_root / "backend" / "models" / "best.pt"
        if not default_path.exists():
            default_path = project_root / "models" / "best.pt"
        weights_path = str(default_path)

    data_yaml = str(project_root / "dataset" / "data.yaml")

    print("=================================================================")
    print("      ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH YOLOV8 - 5 BIỂN BÁO GIAO THÔNG  ")
    print("=================================================================")
    print(f"Trọng số kiểm tra: {weights_path}")
    print(f"Tập dữ liệu:       {split}")
    print(f"Cấu hình:          {data_yaml}")
    print("=================================================================\n")

    model = YOLO(weights_path)

    val_kwargs = {
        "data": data_yaml,
        "split": split,
        "imgsz": imgsz,
        "plots": True,
        "verbose": False
    }
    if device:
        val_kwargs["device"] = device

    t0 = time.time()
    metrics = model.val(**val_kwargs)
    total_time = time.time() - t0

    # Lấy tên các lớp từ model
    names = model.names
    num_classes = len(names)

    # Trích xuất các chỉ số tổng thể
    p_all = metrics.box.mp
    r_all = metrics.box.mr
    map50_all = metrics.box.map50
    map50_95_all = metrics.box.map
    f1_all = 2 * (p_all * r_all) / (p_all + r_all + 1e-16)

    # Tốc độ suy luận (ms / ảnh)
    speed = metrics.speed
    preprocess_ms = speed.get("preprocess", 0.0)
    inference_ms = speed.get("inference", 0.0)
    postprocess_ms = speed.get("postprocess", 0.0)
    total_ms_per_img = preprocess_ms + inference_ms + postprocess_ms

    print("\n" + "=" * 80)
    print("                BẢNG HIỆU SUẤT TỔNG THỂ (OVERALL METRICS)")
    print("=" * 80)
    print(f"{'Chỉ số':<25} | {'Giá trị':<15}")
    print("-" * 80)
    print(f"{'Precision (Độ chính xác)':<25} | {p_all * 100:>6.2f}%")
    print(f"{'Recall (Độ nhạy)':<25} | {r_all * 100:>6.2f}%")
    print(f"{'F1-Score':<25} | {f1_all * 100:>6.2f}%")
    print(f"{'mAP@50':<25} | {map50_all * 100:>6.2f}%")
    print(f"{'mAP@50-95':<25} | {map50_95_all * 100:>6.2f}%")
    print(f"{'Thời gian xử lý / ảnh':<25} | {total_ms_per_img:>6.2f} ms ({inference_ms:.2f} ms inference)")
    print("=" * 80)

    # Trích xuất chỉ số từng class
    print("\n" + "=" * 80)
    print(f"{'Class ID':<8} | {'Mã biển báo':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'mAP@50':<10} | {'mAP@50-95':<10}")
    print("=" * 80)

    p_per_class = metrics.box.p
    r_per_class = metrics.box.r
    map50_per_class = metrics.box.ap50
    map50_95_per_class = metrics.box.ap

    vietnamese_names = {
        "W-208": "Giao nhau với đường ưu tiên",
        "R-303": "Vòng xuyến",
        "P-130": "Cấm dừng và đỗ xe",
        "P-127": "Tốc độ tối đa",
        "P-123b": "Cấm rẽ phải"
    }

    per_class_results = []
    for cls_idx in range(num_classes):
        code = names[cls_idx]
        p = float(p_per_class[cls_idx]) if len(p_per_class) > cls_idx else 0.0
        r = float(r_per_class[cls_idx]) if len(r_per_class) > cls_idx else 0.0
        f1 = 2 * (p * r) / (p + r + 1e-16)
        ap50 = float(map50_per_class[cls_idx]) if len(map50_per_class) > cls_idx else 0.0
        ap = float(map50_95_per_class[cls_idx]) if len(map50_95_per_class) > cls_idx else 0.0

        per_class_results.append({
            "class_id": cls_idx,
            "code": code,
            "name": vietnamese_names.get(code, code),
            "precision": p,
            "recall": r,
            "f1": f1,
            "map50": ap50,
            "map50_95": ap
        })

        print(f"{cls_idx:<8} | {code:<12} | {p*100:>8.2f}% | {r*100:>8.2f}% | {f1*100:>8.2f}% | {ap50*100:>8.2f}% | {ap*100:>8.2f}%")

    print("=" * 80)
    print(f"\n-> Đã lưu Confusion Matrix và các đồ thị trực quan tại: {metrics.save_dir}")

    return {
        "overall": {
            "precision": float(p_all),
            "recall": float(r_all),
            "f1": float(f1_all),
            "map50": float(map50_all),
            "map50_95": float(map50_95_all),
            "inference_time_ms": float(total_ms_per_img)
        },
        "per_class": per_class_results,
        "save_dir": str(metrics.save_dir)
    }

def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình YOLOv8 trên 5 class biển báo")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn file trọng số best.pt")
    parser.add_argument("--split", type=str, default="test", help="Tập đánh giá ('test' hoặc 'val')")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào")
    parser.add_argument("--device", type=str, default="", help="Thiết bị ('0' hoặc 'cpu')")
    args = parser.parse_args()

    evaluate(weights_path=args.weights, split=args.split, imgsz=args.imgsz, device=args.device)

if __name__ == "__main__":
    main()

