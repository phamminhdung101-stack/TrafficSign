# Hệ Thống Nhận Diện Biển Báo Giao Thông Việt Nam (YOLOv8 + FastAPI + Vanilla JS)

Hệ thống ứng dụng AI phát hiện và nhận diện 56 loại biển báo giao thông Việt Nam theo quy chuẩn QCVN 41:2019/BGTVT từ hình ảnh tải lên, quét camera trực tiếp và video.

---

## 1. Yêu cầu môi trường
- **Python**: 3.10 - 3.13 (khuyên dùng môi trường ảo `venv`).
- **MySQL**: 8.0+ (qua XAMPP hoặc MySQL Server / Workbench).
- **Trình duyệt Web** & Visual Studio Code (Live Server extension).

---

## 2. Hướng dẫn chạy dự án từng bước

### Bước 1: Khởi động và chuẩn bị MySQL Database
1. Mở XAMPP hoặc MySQL Workbench, đảm bảo dịch vụ MySQL đang chạy tại cổng `3306`.
2. Chạy file `database/db_traffic.sql` để tạo database `traffic_sign_db` và khởi tạo 56 lớp biển báo cùng bảng dữ liệu người dùng, lịch sử.
3. Tài khoản quản trị mặc định:
   - **Tên đăng nhập**: `admin`
   - **Mật khẩu**: `123456`

### Bước 2: Huấn luyện mô hình YOLOv8 (Nếu muốn train lại)
Dataset đã có sẵn cấu trúc chuẩn tại thư mục `dataset/` (bao gồm `train/`, `valid/`, `test/` và `data.yaml`).
Mở Terminal tại thư mục gốc `d:\traffic-sign-recognition`:

```powershell
# Kích hoạt virtualenv
.\venv\Scripts\Activate.ps1

# Lệnh huấn luyện 50 epoch
yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=50 imgsz=640 batch=16
```

> **Lưu ý quan trọng**: Khi train xong, Ultralytics sẽ lưu trọng số tốt nhất tại `runs/detect/train/weights/best.pt` (hoặc `train-2/weights/best.pt` nếu train lần 2). Hãy copy file `best.pt` này vào `backend/models/best.pt` để ứng dụng tải mô hình mới nhất.

### Bước 3: Chạy Backend (FastAPI)
Mở Terminal tại thư mục `d:\traffic-sign-recognition`:

```powershell
# Kích hoạt môi trường ảo
.\venv\Scripts\Activate.ps1

# Cài đặt thư viện (nếu chưa cài)
pip install -r backend/requirements.txt

# Thiết lập biến môi trường và chạy FastAPI
$env:PYTHONPATH="backend"
uvicorn app.main:app --reload --port 8000 --app-dir backend
```

- API tài liệu Swagger UI: **http://127.0.0.1:8000/docs**
- Kiểm tra trạng thái hệ thống: **http://127.0.0.1:8000/api/health**

### Bước 4: Chạy Frontend bằng Live Server
1. Mở VS Code tại thư mục `frontend/`.
2. Click chuột phải vào file `login.html` (hoặc `upload.html`) chọn **"Open with Live Server"**.
3. Trình duyệt tự động mở tại **http://127.0.0.1:5500/frontend/login.html**.
4. Đăng nhập với tài khoản `admin` / `123456` (hoặc nhấn Đăng ký tài khoản mới).

---

## 3. Các tính năng nổi bật
1. **Tải ảnh**: Hỗ trợ mọi định dạng ảnh JPG, PNG, WEBP, kể cả ảnh chụp góc rộng, ảnh toàn cảnh đường phố hoặc ảnh crop/icon nhỏ (tự động áp dụng Adaptive Context Padding).
2. **Quét Camera**: Chụp ảnh trực tiếp từ webcam trình duyệt và nhận diện tức thì.
3. **Tải Video (MP4/AVI/MOV)**: Tự động trích xuất keyframe, phát hiện biển báo và hiển thị frame tối ưu nhất.
4. **Tùy chỉnh ngưỡng tin cậy**: Thanh trượt điều chỉnh Confidence từ 5% đến 90% ngay trên giao diện upload.
5. **Hiển thị & Tải kết quả**: Bounding box vẽ chính xác trên ảnh kèm mã hiệu và tên tiếng Việt của biển báo (chuẩn QCVN). Nút "Tải ảnh xuống" tự động kết xuất ảnh có bounding box và nhãn nhận diện.
