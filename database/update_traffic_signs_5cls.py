"""
Cập nhật bảng traffic_signs trong MySQL để đồng bộ chính xác 5 class biển báo giao thông:
  Class 0: W-208  - Giao nhau với đường ưu tiên
  Class 1: R-303  - Nơi giao nhau chạy theo vòng xuyến
  Class 2: P-130  - Cấm dừng xe và đỗ xe
  Class 3: P-127  - Tốc độ tối đa cho phép
  Class 4: P-123b - Cấm rẽ phải
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "backend"))

from sqlalchemy import text
from app.services.database import get_engine, DatabaseUnavailableError

TARGET_5_SIGNS = [
    {
        "class_id": 0,
        "code": "W-208",
        "name": "Giao nhau với đường ưu tiên",
        "category": "Biển cảnh báo",
        "description": "Biển số W.208 báo trước sắp đến nơi giao nhau với đường ưu tiên.",
    },
    {
        "class_id": 1,
        "code": "R-303",
        "name": "Nơi giao nhau chạy theo vòng xuyến",
        "category": "Biển hiệu lệnh",
        "description": "Biển số R.303 báo cho các loại xe phải chạy vòng theo đảo an toàn ở nơi đường giao nhau.",
    },
    {
        "class_id": 2,
        "code": "P-130",
        "name": "Cấm dừng xe và đỗ xe",
        "category": "Biển cấm",
        "description": "Biển số P.130 cấm các loại xe cơ giới dừng và đỗ xe ở đoạn đường có đặt biển.",
    },
    {
        "class_id": 3,
        "code": "P-127",
        "name": "Tốc độ tối đa cho phép",
        "category": "Biển cấm",
        "description": "Biển số P.127 cấm các loại xe cơ giới chạy với tốc độ tối đa vượt quá trị số ghi trên biển.",
    },
    {
        "class_id": 4,
        "code": "P-123b",
        "name": "Cấm rẽ phải",
        "category": "Biển cấm",
        "description": "Biển số P.123b cấm các loại xe (cơ giới và thô sơ) rẽ phải ở nơi đường giao nhau.",
    },
]

def update_signs():
    try:
        engine = get_engine()
        with engine.begin() as conn:
            print("Đang cập nhật bảng traffic_signs trong MySQL...")
            # Tạm thời tắt foreign key checks để dọn dẹp hoặc cập nhật an toàn
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
            
            # Xóa các biển báo cũ không thuộc 5 class
            conn.execute(text("DELETE FROM detection_details WHERE traffic_sign_id NOT IN (SELECT id FROM traffic_signs WHERE code IN ('W-208', 'R-303', 'P-130', 'P-127', 'P-123b'));"))
            conn.execute(text("DELETE FROM traffic_signs;"))

            # Thêm mới chính xác 5 class
            for sign in TARGET_5_SIGNS:
                conn.execute(
                    text(
                        "INSERT INTO traffic_signs (class_id, code, name, category, description) "
                        "VALUES (:class_id, :code, :name, :category, :description)"
                    ),
                    sign,
                )

            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
            print("-> Đã đồng bộ 5 biển báo vào MySQL thành công:")
            rows = conn.execute(text("SELECT class_id, code, name FROM traffic_signs ORDER BY class_id;")).fetchall()
            for r in rows:
                print(f"   Class {r[0]}: {r[1]} - {r[2]}")
    except DatabaseUnavailableError as exc:
        print(f"Cảnh báo: Không thể kết nối MySQL (Có thể MySQL chưa bật): {exc}")
    except Exception as exc:
        print(f"Lỗi cập nhật bảng traffic_signs: {exc}")

if __name__ == "__main__":
    update_signs()

