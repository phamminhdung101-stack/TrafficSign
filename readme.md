Bước 1: Thiết lập Database

Mở MySQL (XAMPP hoặc MySQL Workbench) và import file database/db_traffic.sql.
Mở file backend/.env (nếu chưa có thì copy từ .env.example) và trỏ DATABASE_URL về thông tin DB của bạn: DATABASE_URL=mysql+pymysql://root:@127.0.0.1:3306/traffic_sign_db?charset=utf8mb4

Bước 2: Train mô hình YOLOv8 Vì dataset đã có sẵn ở mục dataset/, mở terminal ở thư mục gốc của dự án (d:\traffic-sign-recognition) và chạy lệnh:

pip install ultralytics
yolo task=detect mode=train model=yolov8n.pt data=dataset/data.yaml epochs=50 imgsz=640

Sau khi train xong, copy file trọng số tốt nhất runs/detect/train/weights/best.pt và dán vào thư mục backend/models/best.pt.

Bước 3: Chạy Backend (FastAPI) Mở terminal mới ở thư mục gốc:
# Kích hoạt môi trường ảo (nếu có)
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --reload

API sẽ chạy tại: http://127.0.0.1:8000 (Có thể xem tài liệu API tại http://127.0.0.1:8000/docs).

Bước 4: Chạy Frontend
Bạn có thể cài extension Live Server trên VSCode và nhấn "Go Live" tại file frontend/login.html.
Hoặc đơn giản là click đúp mở file frontend/login.html bằng Chrome/Edge. Đăng ký 1 tài khoản mới và bắt đầu trải nghiệm!