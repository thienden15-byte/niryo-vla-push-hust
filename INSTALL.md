# INSTALL

File này hướng dẫn cài đặt môi trường cho repository.

Project có hai nhóm môi trường chính:

1. ROS 2 / MoveIt cho baseline AI vision + MoveIt
2. Python / PyTorch / PyNiryo cho Mini-VLA, Tiny-VLA, HSV waypoint và rollout robot thật

Hai môi trường này nên được dùng tách biệt để tránh lỗi dependency.

## 1. Môi trường ROS 2 / MoveIt

Môi trường này dùng cho nhánh baseline MoveIt.

Chỉ source ROS 2 khi chạy các node hoặc script liên quan đến ROS 2 / MoveIt.

Lệnh thường dùng:

source /opt/ros/humble/setup.bash

Nếu có workspace ROS 2 riêng, cần source thêm install/setup.bash của workspace đó.

Không nên dùng terminal đã source ROS 2 để train hoặc inference PyTorch nếu không cần thiết.

## 2. Môi trường Python cho Mini-VLA / Tiny-VLA

Môi trường này dùng cho:

- thu thập dữ liệu bằng Python
- xử lý dataset
- train Mini-VLA
- eval offline
- rollout qua PyNiryo
- HSV-assisted waypoint
- Tiny-VLA runtime

Tạo virtual environment:

python3 -m venv .venv

Kích hoạt môi trường:

source .venv/bin/activate

Cài thư viện:

pip install --upgrade pip
pip install -r requirements.txt

## 3. Cấu hình robot và camera

Không hardcode IP robot hoặc camera index trong code.

Tạo file cấu hình local từ file mẫu:

cp .env.example .env

Sau đó sửa các thông số trong .env nếu cần:

- NIRYO_IP
- CAMERA_INDEX
- ROBOT_VELOCITY
- MINIVLA_STEPS
- MINIVLA_ACTION_SCALE
- MINIVLA_MAX_DELTA

File .env thật không được commit lên GitHub.

## 4. Ghi chú quan trọng

Nếu gặp lỗi thư viện khi chạy PyTorch, kiểm tra xem terminal có đang source ROS 2 hay không.

Nếu gặp lỗi ROS 2, kiểm tra xem terminal đã source /opt/ros/humble/setup.bash hay chưa.

Nên dùng hai terminal riêng:

- Terminal A: ROS 2 / MoveIt
- Terminal B: Python / PyTorch / PyNiryo
