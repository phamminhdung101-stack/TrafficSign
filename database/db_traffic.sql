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

INSERT INTO users 
(username, email, password_hash, full_name, role, is_active)
VALUES 
('admin', 'admin@gmail.com', '123456', 'Administrator', 'admin', TRUE);

-- 6. Dữ liệu khởi tạo cho bảng traffic_signs (Tạo tự động từ dataset)
INSERT INTO traffic_signs (class_id, code, name, category, description, image_path) VALUES
(0, 'DP-135', 'Hết tất cả các lệnh cấm', 'Biển báo hết cấm', 'Hình ảnh và nhận diện cho biển báo Hết tất cả các lệnh cấm (DP-135)', NULL),
(1, 'P-102', 'Cấm đi ngược chiều', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm đi ngược chiều (P-102)', NULL),
(2, 'P-103a', 'Cấm xe ô tô', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm xe ô tô (P-103a)', NULL),
(3, 'P-103b', 'Cấm xe ô tô rẽ phải', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm xe ô tô rẽ phải (P-103b)', NULL),
(4, 'P-103c', 'Cấm xe ô tô rẽ trái', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm xe ô tô rẽ trái (P-103c)', NULL),
(5, 'P-104', 'Cấm xe máy', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm xe máy (P-104)', NULL),
(6, 'P-106a', 'Cấm xe ô tô tải', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm xe ô tô tải (P-106a)', NULL),
(7, 'P-106b', 'Cấm xe ô tô tải (theo trọng lượng)', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm xe ô tô tải (theo trọng lượng) (P-106b)', NULL),
(8, 'P-107a', 'Cấm xe ô tô khách và xe ô tô tải', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm xe ô tô khách và xe ô tô tải (P-107a)', NULL),
(9, 'P-112', 'Cấm người đi bộ', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm người đi bộ (P-112)', NULL),
(10, 'P-115', 'Hạn chế trọng lượng xe', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Hạn chế trọng lượng xe (P-115)', NULL),
(11, 'P-117', 'Hạn chế chiều cao', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Hạn chế chiều cao (P-117)', NULL),
(12, 'P-123a', 'Cấm rẽ trái', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm rẽ trái (P-123a)', NULL),
(13, 'P-123b', 'Cấm rẽ phải', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm rẽ phải (P-123b)', NULL),
(14, 'P-124a', 'Cấm quay đầu xe', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm quay đầu xe (P-124a)', NULL),
(15, 'P-124b', 'Cấm xe ô tô quay đầu', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm xe ô tô quay đầu (P-124b)', NULL),
(16, 'P-124c', 'Cấm rẽ trái và quay đầu xe', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm rẽ trái và quay đầu xe (P-124c)', NULL),
(17, 'P-127', 'Tốc độ tối đa cho phép', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Tốc độ tối đa cho phép (P-127)', NULL),
(18, 'P-128', 'Cấm sử dụng còi', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm sử dụng còi (P-128)', NULL),
(19, 'P-130', 'Cấm dừng xe và đỗ xe', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm dừng xe và đỗ xe (P-130)', NULL),
(20, 'P-131a', 'Cấm đỗ xe', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm đỗ xe (P-131a)', NULL),
(21, 'P-137', 'Cấm rẽ trái và rẽ phải', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Cấm rẽ trái và rẽ phải (P-137)', NULL),
(22, 'P-245a', 'Đi chậm', 'Biển cấm', 'Hình ảnh và nhận diện cho biển báo Đi chậm (P-245a)', NULL),
(23, 'R-301c', 'Các xe chỉ được rẽ trái', 'Biển hiệu lệnh và chỉ dẫn', 'Hình ảnh và nhận diện cho biển báo Các xe chỉ được rẽ trái (R-301c)', NULL),
(24, 'R-301d', 'Các xe chỉ được rẽ phải', 'Biển hiệu lệnh và chỉ dẫn', 'Hình ảnh và nhận diện cho biển báo Các xe chỉ được rẽ phải (R-301d)', NULL),
(25, 'R-301e', 'Các xe chỉ được rẽ trái', 'Biển hiệu lệnh và chỉ dẫn', 'Hình ảnh và nhận diện cho biển báo Các xe chỉ được rẽ trái (R-301e)', NULL),
(26, 'R-302a', 'Hướng phải đi vòng chướng ngại vật sang phải', 'Biển hiệu lệnh và chỉ dẫn', 'Hình ảnh và nhận diện cho biển báo Hướng phải đi vòng chướng ngại vật sang phải (R-302a)', NULL),
(27, 'R-302b', 'Hướng phải đi vòng chướng ngại vật sang trái', 'Biển hiệu lệnh và chỉ dẫn', 'Hình ảnh và nhận diện cho biển báo Hướng phải đi vòng chướng ngại vật sang trái (R-302b)', NULL),
(28, 'R-303', 'Nơi giao nhau chạy theo vòng xuyến', 'Biển hiệu lệnh và chỉ dẫn', 'Hình ảnh và nhận diện cho biển báo Nơi giao nhau chạy theo vòng xuyến (R-303)', NULL),
(29, 'R-407a', 'Đường một chiều', 'Biển hiệu lệnh và chỉ dẫn', 'Hình ảnh và nhận diện cho biển báo Đường một chiều (R-407a)', NULL),
(30, 'R-409', 'Chỗ quay xe', 'Biển hiệu lệnh và chỉ dẫn', 'Hình ảnh và nhận diện cho biển báo Chỗ quay xe (R-409)', NULL),
(31, 'R-425', 'Bệnh viện', 'Biển hiệu lệnh và chỉ dẫn', 'Hình ảnh và nhận diện cho biển báo Bệnh viện (R-425)', NULL),
(32, 'R-434', 'Bến xe buýt', 'Biển hiệu lệnh và chỉ dẫn', 'Hình ảnh và nhận diện cho biển báo Bến xe buýt (R-434)', NULL),
(33, 'S-509a', 'Thuyết minh biển chính', 'Biển phụ', 'Hình ảnh và nhận diện cho biển báo Thuyết minh biển chính (S-509a)', NULL),
(34, 'W-201a', 'Chỗ ngoặt nguy hiểm vòng bên trái', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Chỗ ngoặt nguy hiểm vòng bên trái (W-201a)', NULL),
(35, 'W-201b', 'Chỗ ngoặt nguy hiểm vòng bên phải', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Chỗ ngoặt nguy hiểm vòng bên phải (W-201b)', NULL),
(36, 'W-202a', 'Nhiều chỗ ngoặt nguy hiểm liên tiếp', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Nhiều chỗ ngoặt nguy hiểm liên tiếp (W-202a)', NULL),
(37, 'W-202b', 'Nhiều chỗ ngoặt nguy hiểm liên tiếp', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Nhiều chỗ ngoặt nguy hiểm liên tiếp (W-202b)', NULL),
(38, 'W-203b', 'Đường hẹp bên trái', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Đường hẹp bên trái (W-203b)', NULL),
(39, 'W-203c', 'Đường hẹp bên phải', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Đường hẹp bên phải (W-203c)', NULL),
(40, 'W-205a', 'Đường giao nhau', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Đường giao nhau (W-205a)', NULL),
(41, 'W-205b', 'Đường giao nhau', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Đường giao nhau (W-205b)', NULL),
(42, 'W-205d', 'Đường giao nhau', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Đường giao nhau (W-205d)', NULL),
(43, 'W-207a', 'Giao nhau với đường không ưu tiên', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Giao nhau với đường không ưu tiên (W-207a)', NULL),
(44, 'W-207b', 'Giao nhau với đường không ưu tiên', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Giao nhau với đường không ưu tiên (W-207b)', NULL),
(45, 'W-207c', 'Giao nhau với đường không ưu tiên', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Giao nhau với đường không ưu tiên (W-207c)', NULL),
(46, 'W-208', 'Giao nhau với đường ưu tiên', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Giao nhau với đường ưu tiên (W-208)', NULL),
(47, 'W-209', 'Giao nhau có tín hiệu đèn', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Giao nhau có tín hiệu đèn (W-209)', NULL),
(48, 'W-210', 'Giao nhau với đường sắt có rào chắn', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Giao nhau với đường sắt có rào chắn (W-210)', NULL),
(49, 'W-219', 'Dốc xuống nguy hiểm', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Dốc xuống nguy hiểm (W-219)', NULL),
(50, 'W-224', 'Đường người đi bộ cắt ngang', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Đường người đi bộ cắt ngang (W-224)', NULL),
(51, 'W-225', 'Trẻ em', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Trẻ em (W-225)', NULL),
(52, 'W-227', 'Công trường', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Công trường (W-227)', NULL),
(53, 'W-233', 'Nguy hiểm khác', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Nguy hiểm khác (W-233)', NULL),
(54, 'W-235', 'Đường đôi', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Đường đôi (W-235)', NULL),
(55, 'W-245a', 'Đi chậm', 'Biển báo nguy hiểm', 'Hình ảnh và nhận diện cho biển báo Đi chậm (W-245a)', NULL);
