# Hệ Thống Nhận Diện 5 Biển Báo Giao Thông Việt Nam (YOLOv8 + FastAPI + Vanilla JS)

Hệ thống ứng dụng Computer Vision và Deep Learning (YOLOv8) để nhận diện chính xác **5 loại biển báo giao thông Việt Nam** theo Quy chuẩn QCVN 41:2019/BGTVT từ ảnh chụp, webcam trực tiếp và video.

---

## 1. Giới hạn 5 Class Nhận Diện

| Class ID | Mã biển báo | Tên biển báo tiếng Việt | Nhóm biển báo |
|:---:|:---:|:---|:---|
| **0** | **W-208** | Giao nhau với đường ưu tiên | Biển cảnh báo |
| **1** | **R-303** | Nơi giao nhau chạy theo vòng xuyến | Biển hiệu lệnh |
| **2** | **P-130** | Cấm dừng xe và đỗ xe | Biển cấm |
| **3** | **P-127** | Tốc độ tối đa cho phép | Biển cấm |
| **4** | **P-123b** | Cấm rẽ phải | Biển cấm |

---

## 2. Cấu Trúc Dự Án Chuẩn Hóa

```text
d:\traffic-sign-recognition\
├── dataset/                  # Dữ liệu ảnh và nhãn chuẩn hóa YOLOv8
│   ├── images/               # Thư mục ảnh (train, val, test)
│   ├── labels/               # Thư mục nhãn bounding box (train, val, test)
│   └── data.yaml             # Cấu hình dataset 5 class
├── models/                   # Trọng số mô hình YOLOv8 (.pt)
│   └── best.pt               # Trọng số tốt nhất (mAP50 = 96.30%)
├── training/                 # Mã nguồn tiền xử lý và huấn luyện
│   ├── inspect_dataset.py    # Kiểm tra phân bố và tính toàn vẹn dataset
│   ├── prepare_dataset.py    # Lọc & chuyển đổi dataset 56 classes -> 5 classes
│   └── train.py              # Script huấn luyện YOLOv8 (tối ưu cho biển báo)
├── evaluation/               # Mã nguồn đánh giá mô hình
│   └── evaluate.py           # Đánh giá Precision, Recall, F1, mAP50, mAP50-95
├── tests/                    # Bộ kiểm thử tự động
│   └── test_5_levels.py      # Kiểm thử 4 cấp độ (Cơ bản, Trung bình, Nâng cao, Âm tính)
├── backend/                  # API Backend (FastAPI + SQLAlchemy + MySQL)
│   ├── app/
│   │   ├── main.py           # FastAPI endpoints (/api/detect, /api/signs,...)
│   │   ├── config.py         # Cấu hình hệ thống & database
│   │   ├── schemas.py        # Pydantic schemas
│   │   └── services/         # Detector, Database, History, User Authentication
│   └── tests/                # Unit tests cho Backend
├── frontend/                 # Giao diện Website (HTML/CSS/JS)
│   ├── login.html            # Đăng nhập
│   ├── register.html         # Đăng ký tài khoản
│   ├── upload.html           # Tải ảnh, video & Quét Camera trực tiếp
│   ├── result.html           # Hiển thị bounding box & thông tin biển báo
│   └── history.html          # Lịch sử các lần nhận diện
└── database/                 # Cấu hình cơ sở dữ liệu
    ├── db_traffic.sql        # Script tạo bảng và seed 5 class vào MySQL
    └── update_traffic_signs_5cls.py # Script đồng bộ 5 class vào MySQL
```

---

## 3. Hướng Dẫn Cài Đặt Môi Trường (Windows & PyCharm)

### Bước 1: Cài đặt Python & Virtual Environment
1. Đảm bảo máy đã cài đặt **Python 3.10 - 3.13**.
2. Mở Terminal (PowerShell hoặc Command Prompt) tại thư mục dự án `d:\traffic-sign-recognition`:
```powershell
# Tạo môi trường ảo (nếu chưa có)
python -m venv venv

# Kích hoạt môi trường ảo
# Trên PowerShell:
.\venv\Scripts\Activate.ps1
# Trên Command Prompt (CMD):
venv\Scripts\activate.bat
```

### Bước 2: Cài đặt thư viện phụ thuộc
```powershell
pip install -r backend/requirements.txt
```
*(Nếu máy có GPU NVIDIA, hệ thống tự động tận dụng CUDA với PyTorch `cu124` để tăng tốc độ gấp 15 lần).*

### Bước 3: Cấu hình trên PyCharm (Tùy chọn)
1. Mở PyCharm $\rightarrow$ Chọn **File $\rightarrow$ Open** $\rightarrow$ Trỏ tới `d:\traffic-sign-recognition`.
2. Vào **File $\rightarrow$ Settings $\rightarrow$ Project: traffic-sign-recognition $\rightarrow$ Python Interpreter**.
3. Bấm **Add Interpreter $\rightarrow$ Existing Environment** $\rightarrow$ Trỏ tới `d:\traffic-sign-recognition\venv\Scripts\python.exe`.
4. Nhấn **Apply** và **OK**.

---

## 4. Hướng Dẫn Huấn Luyện & Đánh Giá Mô Hình

### 1. Kiểm tra và chuẩn bị Dataset 5 Class
```powershell
# Kiểm tra số lượng ảnh và tính toàn vẹn
python training/inspect_dataset.py

# Lọc và tạo cấu trúc dataset chuẩn hóa
python training/prepare_dataset.py
```

### 2. Huấn luyện mô hình YOLOv8
```powershell
# Huấn luyện mô hình với GPU (device=0) hoặc CPU (device=cpu)
python train.py --epochs 30 --batch 16 --imgsz 640 --device 0
```
> **Đặc điểm tối ưu**: Script tự động tắt lật ngang (`fliplr=0.0`) để tránh làm biến dạng biển chỉ hướng `P-123b` (Cấm rẽ phải) và `R-303` (Vòng xuyến). Trọng số tốt nhất tự động được lưu vào `models/best.pt` và `backend/models/best.pt`.

### 3. Đánh giá mô hình trên tập Test
```powershell
python evaluation.py --split test
```

### 4. Chạy bộ kiểm thử tự động 4 cấp độ (Levels 1 - 4)
```powershell
python tests/test_5_levels.py
```

---

## 5. Hướng Dẫn Khởi Chạy Dự Án

### Bước 1: Khởi động MySQL Database
1. Mở XAMPP / MySQL Workbench, đảm bảo dịch vụ MySQL đang chạy tại cổng `3306`.
2. Đồng bộ 5 class vào cơ sở dữ liệu:
```powershell
$env:PYTHONPATH="backend"
python database/update_traffic_signs_5cls.py
```

### Bước 2: Khởi động Backend (FastAPI)
```powershell
$env:PYTHONPATH="backend"
uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir backend --reload
```
- API Health Check: `http://127.0.0.1:8000/api/health`
- Swagger UI Docs: `http://127.0.0.1:8000/docs`

### Bước 3: Khởi động Frontend (Live Server)
1. Trong VS Code, chuột phải vào [`frontend/login.html`](file:///d:/traffic-sign-recognition/frontend/login.html) $\rightarrow$ chọn **Open with Live Server**.
2. Trình duyệt tự động mở tại `http://127.0.0.1:5500/frontend/login.html`.
3. Đăng nhập với tài khoản:
   - **Tài khoản**: `admin`
   - **Mật khẩu**: `admin123` *(hoặc `123456`, hoặc tự đăng ký tài khoản mới)*.
