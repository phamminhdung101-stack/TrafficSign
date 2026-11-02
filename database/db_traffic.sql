CREATE DATABASE IF NOT EXISTS traffic_sign_db
DEFAULT CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE traffic_sign_db;

-- 1. Bảng lưu trữ Người dùng (Thêm mới cho User / Admin)
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL, -- Cần dài 255 ký tự để lưu chuỗi băm (vd: bcrypt)
    full_name VARCHAR(100),
    role ENUM('user', 'admin') NOT NULL DEFAULT 'user', -- Phân quyền trực tiếp tại đây
    is_active BOOLEAN NOT NULL DEFAULT TRUE, -- Admin có thể khóa tài khoản User nếu cần
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. Bảng thông tin Biển báo giao thông
CREATE TABLE traffic_signs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    class_id INT NOT NULL UNIQUE,
    code VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    description TEXT,
    image_path VARCHAR(255),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. Bảng Quản lý Mô hình AI (YOLO)
CREATE TABLE models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50),
    model_path VARCHAR(255) NOT NULL,
    epochs INT,
    map50 FLOAT,
    map50_95 FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    is_active BOOLEAN NOT NULL DEFAULT FALSE, -- Trạng thái để biết mô hình nào đang được chọn chạy chính
    created_by INT, -- Theo dõi Admin nào đã upload/huấn luyện mô hình này
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_models_created_by FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4. Bảng Lịch sử Nhận diện
CREATE TABLE detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL, -- Bắt buộc: Cần biết User nào đã thực hiện nhận diện
    model_id INT,
    source_type ENUM('image', 'video', 'camera') NOT NULL, -- Thay VARCHAR bằng ENUM để dữ liệu chặt chẽ hơn
    source_path VARCHAR(255),
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_detections_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_detections_model_id FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. Bảng Chi tiết Kết quả Nhận diện (Bounding Boxes)
CREATE TABLE detection_details (
    id INT AUTO_INCREMENT PRIMARY KEY,
    detection_id INT NOT NULL,
    traffic_sign_id INT NOT NULL,
    confidence FLOAT NOT NULL,
    x_center FLOAT,
    y_center FLOAT,
    width FLOAT,
    height FLOAT,
    
    CONSTRAINT chk_confidence CHECK (confidence >= 0 AND confidence <= 1), -- Đảm bảo độ tin cậy luôn từ 0-1
    CONSTRAINT fk_details_detection_id FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE,
    CONSTRAINT fk_details_traffic_sign_id FOREIGN KEY (traffic_sign_id) REFERENCES traffic_signs(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 6. Dữ liệu khởi tạo cho bảng traffic_signs (5 Biển báo giao thông Việt Nam mục tiêu)
INSERT INTO traffic_signs (class_id, code, name, category, description, image_path) VALUES
(0, 'W-208', 'Giao nhau với đường ưu tiên', 'Biển cảnh báo', 'Biển số W.208 báo trước sắp đến nơi giao nhau với đường ưu tiên.', NULL),
(1, 'R-303', 'Nơi giao nhau chạy theo vòng xuyến', 'Biển hiệu lệnh', 'Biển số R.303 báo cho các loại xe phải chạy vòng theo đảo an toàn ở nơi đường giao nhau.', NULL),
(2, 'P-130', 'Cấm dừng xe và đỗ xe', 'Biển cấm', 'Biển số P.130 cấm các loại xe cơ giới dừng và đỗ xe ở đoạn đường có đặt biển.', NULL),
(3, 'P-127', 'Tốc độ tối đa cho phép', 'Biển cấm', 'Biển số P.127 cấm các loại xe cơ giới chạy với tốc độ tối đa vượt quá trị số ghi trên biển.', NULL),
(4, 'P-123b', 'Cấm rẽ phải', 'Biển cấm', 'Biển số P.123b cấm các loại xe (cơ giới và thô sơ) rẽ phải ở nơi đường giao nhau.', NULL);