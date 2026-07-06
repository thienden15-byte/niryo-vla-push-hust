# 01. Cài đặt môi trường

File này hướng dẫn chuẩn bị môi trường cơ bản để tiếp tục phát triển project Niryo VLA Push.

Project có hai nhóm môi trường chính:

1. ROS 2 / MoveIt
2. Python / PyTorch / PyNiryo

Hai nhóm môi trường này nên được dùng tách biệt để tránh lỗi dependency.

---

## 1. Môi trường ROS 2 / MoveIt

Môi trường ROS 2 / MoveIt dùng cho nhánh baseline AI vision + MoveIt.

Nhánh này có thể bao gồm:

- ROS 2 nodes
- custom message
- MoveIt planning
- RViz visualization
- launch files
- workspace catkin hoặc colcon

Lệnh thường dùng khi chạy ROS 2 / MoveIt:

source /opt/ros/humble/setup.bash

Sau đó, nếu project có workspace ROS 2 riêng, cần source thêm workspace đó. Ví dụ:

source ~/niryo_ws/install/setup.bash

Lưu ý:

- Chỉ source ROS 2 khi chạy phần ROS 2 / MoveIt.
- Không nên source ROS 2 trong terminal dùng để train Mini-VLA hoặc Tiny-VLA bằng PyTorch.
- Không commit các thư mục build/, install/, log/ của ROS 2 lên GitHub.

---

## 2. Môi trường Python / PyTorch / PyNiryo

Môi trường Python dùng cho các phần:

- thu thập dữ liệu robot thật
- xử lý dataset
- train Mini-VLA
- eval offline
- rollout Direct Mini-VLA trên robot thật
- HSV-assisted waypoint
- Tiny-VLA runtime
- PyNiryo robot control

Tạo virtual environment:

python3 -m venv .venv

Kích hoạt môi trường:

source .venv/bin/activate

Cài thư viện:

pip install --upgrade pip
pip install -r requirements.txt

Kiểm tra Python đang dùng:

which python
python --version

Kiểm tra pip đang dùng:

which pip
pip --version

---

## 3. Không trộn terminal ROS 2 và PyTorch

Nên dùng hai terminal riêng.

Terminal A dùng cho ROS 2 / MoveIt:

source /opt/ros/humble/setup.bash
source ~/niryo_ws/install/setup.bash

Terminal B dùng cho Python / PyTorch / PyNiryo:

source .venv/bin/activate

Lý do:

- ROS 2 có nhiều biến môi trường riêng.
- PyTorch, OpenCV, NumPy có thể bị ảnh hưởng bởi môi trường đã source ROS 2.
- Tách terminal giúp dễ debug hơn.

---

## 4. Cấu hình robot và camera

Repo dùng file mẫu:

.env.example

Người dùng tạo file cấu hình thật bằng lệnh:

cp .env.example .env

Sau đó sửa các thông số trong .env theo máy thật:

- NIRYO_IP
- CAMERA_INDEX
- ROBOT_VELOCITY
- MINIVLA_STEPS
- MINIVLA_ACTION_SCALE
- MINIVLA_MAX_DELTA

File .env chứa cấu hình local nên không commit lên GitHub.

---

## 5. Kiểm tra nhanh môi trường Python

Sau khi cài xong, có thể kiểm tra các thư viện chính:

python -c "import numpy; print('numpy ok')"
python -c "import cv2; print('opencv ok')"
python -c "import torch; print('torch ok')"
python -c "import pyniryo; print('pyniryo ok')"

Nếu import torch lỗi, kiểm tra lại môi trường Python.

Nếu import pyniryo lỗi, kiểm tra lại requirements.txt hoặc cài PyNiryo theo hướng dẫn chính thức.

---

## 6. Ghi chú cho người phát triển sau

Trước khi chạy robot thật, cần đọc tiếp:

- 02_ket_noi_robot_camera.md
- 03_cau_hinh_env.md
- 04_loi_thuong_gap.md

Không chạy script điều khiển robot thật khi chưa kiểm tra:

- robot đã kết nối đúng IP
- camera đã mở đúng index
- workspace không có vật cản
- tốc độ robot đang ở mức an toàn
- action scale và max delta đã được giới hạn
