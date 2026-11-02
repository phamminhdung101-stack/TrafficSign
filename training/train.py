import argparse
import os
import shutil
from pathlib import Path
import cv2

# Ngăn chặn xung đột luồng OpenCV và OpenMP trên môi trường Windows
cv2.setNumThreads(0)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình YOLOv8 nhận diện 5 biển báo giao thông Việt Nam")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Trọng số khởi tạo pretrained (yolov8n.pt)")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Đường dẫn file cấu hình data.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="Số lượng epochs huấn luyện (mặc định 30)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (mặc định 16; giảm xuống 8 nếu VRAM yếu)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào (mặc định 640)")
    parser.add_argument("--device", type=str, default="0", help="Thiết bị huấn luyện ('0' cho GPU, 'cpu' cho CPU)")
    parser.add_argument("--workers", type=int, default=0, help="Số workers DataLoader (mặc định 0: tối ưu nhất trên Windows, tránh lỗi _try_get_data)")
    parser.add_argument("--project", type=str, default="runs/detect", help="Thư mục lưu kết quả huấn luyện")
    parser.add_argument("--name", type=str, default="train_5cls", help="Tên phiên huấn luyện")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_yaml = project_root / args.data
    weights = project_root / args.model

    if not data_yaml.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình dataset: {data_yaml}")
    if not weights.exists():
        raise FileNotFoundError(f"Không tìm thấy file trọng số khởi tạo: {weights}")

    print("=================================================================")
    print("      QUÁ TRÌNH HUẤN LUYỆN YOLOV8 - 5 BIỂN BÁO GIAO THÔNG       ")
    print("=================================================================")
    print(f"Model Pretrained: {weights}")
    print(f"Dataset Config:   {data_yaml}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size:       {args.batch}")
    print(f"Image Size:       {args.imgsz}")
    print(f"Workers:          {args.workers} (workers=0 loại bỏ hoàn toàn lỗi multiprocessing trên Windows)")
    print(f"Device:           {'auto' if not args.device else args.device}")
    print("-----------------------------------------------------------------")
    print("Lưu ý về Data Augmentation biển báo:")
    print("  - fliplr=0.0: Không lật ngang vì biển P-123b (Cấm rẽ phải) và")
    print("    R-303 (Vòng xuyến) mang đặc trưng định hướng chiều di chuyển.")
    print("=================================================================\n")

    model = YOLO(str(weights))

    # Huấn luyện mô hình với cấu hình tối ưu và bảo vệ DataLoader trên Windows
    train_args = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "project": str(project_root / args.project),
        "name": args.name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "plots": True,
        "save": True,
        "amp": False,    # Tắt AMP check để tránh tải yolo26n.pt từ GitHub bị nghẽn mạng
        # Augmentation settings customized for directional traffic signs:
        "fliplr": 0.0,   # Tuyệt đối không lật ngang tránh biến cấm rẽ phải thành cấm rẽ trái
        "flipud": 0.0,   # Không lật dọc
        "degrees": 10.0, # Nghiêng nhẹ +/- 10 độ mô phỏng camera thực tế
        "shear": 2.0,
        "perspective": 0.0005,
        "scale": 0.3,    # Co giãn kích thước để nhận diện biển xa và gần
        "mosaic": 1.0,   # Ghép 4 ảnh tăng khả năng phát hiện đa kích thước
    }

    if args.device:
        train_args["device"] = args.device

    results = model.train(**train_args)

    # Đường dẫn file trọng số tốt nhất sau train
    exp_dir = project_root / args.project / args.name
    best_weight = exp_dir / "weights" / "best.pt"
    last_weight = exp_dir / "weights" / "last.pt"

    if best_weight.exists():
        print("\n=================================================================")
        print("                HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!                 ")
        print("=================================================================")
        print(f"Trọng số tốt nhất (best.pt): {best_weight}")
        print(f"Trọng số cuối cùng (last.pt): {last_weight}")

        # Đồng bộ trọng số sang thư mục models/ và backend/models/
        models_dir = project_root / "models"
        backend_models_dir = project_root / "backend" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        backend_models_dir.mkdir(parents=True, exist_ok=True)

        target_best_1 = models_dir / "best.pt"
        target_best_2 = backend_models_dir / "best.pt"

        shutil.copy2(best_weight, target_best_1)
        shutil.copy2(best_weight, target_best_2)
        print(f"-> Đã sao chép best.pt sang: {target_best_1}")
        print(f"-> Đã sao chép best.pt sang: {target_best_2}")
    else:
        print("Cảnh báo: Không tìm thấy file best.pt sau khi train.")

if __name__ == "__main__":
    main()

