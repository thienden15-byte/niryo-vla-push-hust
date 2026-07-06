# 02. Kết nối robot và camera

File này hướng dẫn kiểm tra kết nối robot Niryo Ned và camera trước khi chạy thu dữ liệu, rollout hoặc HSV-assisted waypoint.

Không chạy robot thật nếu chưa kiểm tra kết nối, camera, workspace và tốc độ an toàn.

---

## 1. Thông tin kết nối hiện tại

Thông tin hiện tại của hệ thống:

- Robot IP: 169.254.200.200
- Kiểu kết nối: nhiều khả năng là Direct LAN / Ethernet link-local connection
- PyNiryo runtime: không cần source ROS 2 Humble
- Camera thường dùng: USB camera IMX179
- Camera index thường gặp: /dev/video3
- Camera name thường gặp: USB Camera2 hoặc USB Camera2: USB Camera2

Lưu ý:

- IP 169.254.x.x là dải link-local, thường xuất hiện khi kết nối trực tiếp qua Ethernet.
- Kiểu kết nối LAN trực tiếp cần xác nhận lại bằng ip addr, ip route hoặc nmcli.
- Camera index có thể thay đổi sau khi rút/cắm lại camera hoặc khởi động lại máy.
- Không nên hardcode camera index nếu có thể auto-detect theo tên camera.

---

## 2. Kiểm tra mạng tới robot

Kiểm tra robot có phản hồi trong mạng hay không:

ping -c 3 169.254.200.200

Nếu ping thành công, máy tính có thể nhìn thấy robot.

Nếu ping thất bại, kiểm tra:

- robot đã bật nguồn chưa
- dây LAN hoặc kết nối mạng
- máy tính có cùng dải mạng với robot không
- IP robot có đúng không
- firewall hoặc network manager có chặn không

Kiểm tra cấu hình mạng trên máy:

ip addr
ip route
nmcli device status

Nếu máy có IP cùng dải 169.254.x.x trên cổng Ethernet, có thể ghi nhận là Direct LAN / Ethernet link-local connection.

---

## 3. Kiểm tra PyNiryo

Luồng robot thật hiện tại dùng PyNiryo.

Với PyNiryo runtime, hiện tại không cần source ROS 2 Humble. Chỉ cần môi trường Python có cài pyniryo.

Kiểm tra nhanh PyNiryo:

python3 - <<'PY'
from pyniryo import NiryoRobot

IP = "169.254.200.200"

robot = NiryoRobot(IP)
robot.set_learning_mode(False)

print("Connected to Niryo:", IP)
print("Joints:", robot.get_joints())
print("Pose:", robot.get_pose())

robot.close_connection()
print("Robot safety check OK")
PY

Nếu lỗi kết nối PyNiryo, kiểm tra:

- ping tới robot có thành công không
- IP robot có đúng không
- robot có đang bật không
- robot có đang bị chương trình khác chiếm kết nối không
- môi trường Python có cài pyniryo chưa
- có đang chạy nhầm terminal ROS 2 hay không

---

## 4. Script robot thật hiện có

Các script robot thật hiện có nằm trong:

~/tinyvla_niryo_runtime/scripts

Một số file quan trọng:

~/tinyvla_niryo_runtime/scripts/run_author_style_niryo_diffik_rollout.py
~/tinyvla_niryo_runtime/scripts/run_auto_green_monotonic_push.py
~/tinyvla_niryo_runtime/scripts/contact_descent_diffik_assist.py

Có thể tìm các file dùng PyNiryo bằng lệnh:

grep -R "NiryoRobot\|pyniryo" -n ~/tinyvla_niryo_runtime/scripts ~/TinyVLA 2>/dev/null | head -50

Hiện tại chưa xác nhận có script test PyNiryo riêng. Nếu cần bàn giao sạch hơn, nên tạo script riêng:

~/tinyvla_niryo_runtime/scripts/test_niryo_connection.py

hoặc trong repository này:

scripts/phase_5_real_robot_test/test_niryo_connection.py

---

## 5. Kiểm tra camera USB

Liệt kê các thiết bị video:

ls -l /dev/video*

Kiểm tra tên camera:

v4l2-ctl --list-devices

Nếu chưa có v4l2-ctl, cài bằng:

sudo apt install v4l-utils

Kiểm tra tên camera qua sysfs:

for i in /sys/class/video4linux/video*/name; do
  echo "$i: $(cat $i)"
done

Camera IMX179 từng được nhận với tên chứa:

USB Camera2

Camera thường gặp ở:

/dev/video3

Nhưng không nên ghi cứng tuyệt đối vì camera index có thể đổi.

---

## 6. Kiểm tra camera bằng OpenCV

Test mở camera index 3:

python3 - <<'PY'
import cv2

cam_idx = 3
cap = cv2.VideoCapture(cam_idx)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera /dev/video{cam_idx}")

ok, frame = cap.read()
cap.release()

print("Camera index:", cam_idx)
print("Read OK:", ok)
if ok:
    print("Frame shape:", frame.shape)
PY

Nếu không mở được, thử các index khác:

- 0
- 1
- 2
- 3

Nếu mở được nhưng không đúng camera nhìn workspace, kiểm tra lại bằng v4l2-ctl --list-devices.

---

## 7. Script camera hiện có

Script kiểm tra camera/runtime hiện có:

~/tinyvla_niryo_runtime/scripts/test_author10d_live_camera_only.py

Các script khác cũng có mở camera:

~/tinyvla_niryo_runtime/scripts/run_author_style_niryo_diffik_rollout.py
~/tinyvla_niryo_runtime/scripts/run_auto_green_monotonic_push.py

Tìm các script có dùng OpenCV hoặc camera:

grep -R "cv2.VideoCapture\|find_usb_camera\|LiveCamera" -n ~/tinyvla_niryo_runtime/scripts ~/TinyVLA 2>/dev/null | head -80

---

## 8. Cấu hình IP và camera trong code

Hiện tại các script robot thật chưa dùng .env làm chuẩn chính.

Cách cấu hình hiện tại chủ yếu là:

- argparse default
- hardcode default trong script
- camera auto-detect theo tên USB Camera2

Ví dụ:

- robot IP mặc định: 169.254.200.200
- camera thường auto-detect theo tên USB Camera2
- camera index thường gặp: /dev/video3

Trong repository bàn giao, nên chuẩn hóa dần sang hướng:

- dùng .env.example làm file mẫu
- người dùng copy thành .env
- code đọc NIRYO_IP và CAMERA_INDEX từ .env hoặc command-line argument

Không commit file .env thật lên GitHub.

---

## 9. Checklist trước khi chạy robot thật

Trước khi chạy script có điều khiển robot thật, cần kiểm tra:

- ping được robot
- PyNiryo đọc được joints và pose
- robot không ở trạng thái lỗi
- learning mode được tắt khi cần điều khiển
- robot đang ở vị trí an toàn
- workspace không có vật cản
- gripper không cạ bàn
- object nằm trong vùng làm việc
- camera nhìn rõ object
- camera index đúng
- velocity thấp
- action scale đã giới hạn
- max delta đã giới hạn
- script có chế độ test hoặc dry-run nếu cần

Không nên chạy execute trực tiếp nếu chưa kiểm tra kết nối và quan sát vùng làm việc.

---

## 10. Tóm tắt thông tin hiện tại

Robot IP:

169.254.200.200

Connection type:

Direct LAN / Ethernet link-local connection, cần xác nhận bằng ip addr hoặc nmcli.

PyNiryo test script:

Chưa có script test riêng được xác nhận. Các script PyNiryo chính nằm trong ~/tinyvla_niryo_runtime/scripts.

Need ROS 2 for PyNiryo:

Không. PyNiryo runtime chỉ cần Python environment.

Camera index:

Thường là /dev/video3.

Camera name:

USB Camera2 hoặc USB Camera2: USB Camera2.

Camera test script:

~/tinyvla_niryo_runtime/scripts/test_author10d_live_camera_only.py

Config source:

Hiện chủ yếu dùng argparse/default trong script. NIRYO_IP có default 169.254.200.200. Camera thường auto-detect theo USB Camera2. Chưa dùng .env làm chuẩn chính.

Safety check robot:

ping 169.254.200.200, sau đó PyNiryo get_joints và get_pose.

Safety check camera:

v4l2-ctl --list-devices, kiểm tra /sys/class/video4linux/video*/name, sau đó OpenCV VideoCapture đọc frame.
