# 03. Lỗi thường gặp khi setup và kết nối

File này ghi lại các lỗi thường gặp khi cài đặt môi trường, kết nối robot Niryo Ned và kiểm tra camera.

---

## 1. Không ping được robot

Lệnh kiểm tra:

ping -c 3 169.254.200.200

Nếu không ping được, kiểm tra:

- robot đã bật nguồn chưa
- dây LAN đã cắm chưa
- máy tính có cùng dải mạng với robot không
- IP robot có đúng không
- Network Manager có đang dùng đúng card mạng không

Kiểm tra mạng:

ip addr
ip route
nmcli device status

Nếu dùng IP 169.254.x.x, nhiều khả năng robot đang kết nối theo kiểu Direct LAN / Ethernet link-local connection.

---

## 2. PyNiryo không kết nối được robot

Triệu chứng:

- script bị treo khi tạo NiryoRobot
- báo lỗi connection timeout
- không đọc được joints hoặc pose

Cần kiểm tra:

- ping tới robot có thành công không
- IP robot có đúng là 169.254.200.200 không
- robot có đang bật và sẵn sàng không
- có chương trình khác đang chiếm kết nối robot không
- môi trường Python có cài pyniryo chưa

Kiểm tra nhanh:

python3 - <<'PY'
from pyniryo import NiryoRobot

IP = "169.254.200.200"
robot = NiryoRobot(IP)

print("Joints:", robot.get_joints())
print("Pose:", robot.get_pose())

robot.close_connection()
print("OK")
PY

---

## 3. Nhầm môi trường ROS 2 và PyTorch

Project có hai nhóm môi trường:

- ROS 2 / MoveIt
- Python / PyTorch / PyNiryo

Không nên dùng chung một terminal cho mọi thứ.

Khuyến nghị:

Terminal cho ROS 2 / MoveIt:

source /opt/ros/humble/setup.bash
source ~/niryo_ws/install/setup.bash

Terminal cho Mini-VLA / Tiny-VLA / PyTorch:

source .venv/bin/activate

Nếu PyTorch hoặc OpenCV lỗi bất thường, kiểm tra xem terminal có đang source ROS 2 hay không.

---

## 4. Không thấy camera

Kiểm tra danh sách camera:

ls -l /dev/video*

Kiểm tra tên camera:

v4l2-ctl --list-devices

Nếu thiếu v4l2-ctl, cài:

sudo apt install v4l-utils

Kiểm tra tên camera qua sysfs:

for i in /sys/class/video4linux/video*/name; do
  echo "$i: $(cat $i)"
done

Camera IMX179 trong project thường có tên chứa:

USB Camera2

---

## 5. Sai camera index

Camera thường gặp ở:

/dev/video3

Nhưng index có thể thay đổi sau khi rút/cắm camera hoặc reboot máy.

Test nhanh bằng OpenCV:

python3 - <<'PY'
import cv2

for cam_idx in range(5):
    cap = cv2.VideoCapture(cam_idx)
    ok = cap.isOpened()
    ret, frame = cap.read() if ok else (False, None)
    cap.release()

    print("camera", cam_idx, "opened:", ok, "read:", ret, "shape:", None if frame is None else frame.shape)
PY

Nếu mở được nhưng hình không đúng camera nhìn workspace, cần đổi CAMERA_INDEX hoặc dùng auto-detect theo tên USB Camera2.

---

## 6. Robot di chuyển quá mạnh hoặc không an toàn

Trước khi chạy robot thật, kiểm tra:

- velocity đang thấp
- action scale không quá lớn
- max delta đã giới hạn
- workspace không có vật cản
- gripper không cạ bàn
- object nằm trong vùng làm việc
- người vận hành sẵn sàng dừng robot

Với rollout Mini-VLA, nên test bước nhỏ trước khi chạy nhiều bước.

---

## 7. File .env không được đọc

Trạng thái hiện tại của một số script cũ:

- chưa dùng .env làm chuẩn chính
- dùng argparse default
- dùng hardcode default
- camera có thể auto-detect theo USB Camera2

Vì vậy khi chạy script, cần đọc kỹ help hoặc code của script đó.

Kiểm tra tham số script:

python3 path/to/script.py --help

Về lâu dài, nên chuẩn hóa script theo hướng:

- đọc .env nếu có
- cho phép override bằng command-line argument
- không hardcode trực tiếp trong code

---

## 8. Không nên commit file lớn

Không commit trực tiếp các file:

- .npz
- .hdf5
- .h5
- .pt
- .pth
- .ckpt
- .mp4
- .avi
- .env

Các file lớn nên đưa lên Google Drive, GitHub Release hoặc nền tảng lưu trữ ngoài, rồi ghi link trong datasets/ hoặc checkpoints/.
