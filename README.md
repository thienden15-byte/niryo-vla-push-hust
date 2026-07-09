# Niryo VLA Push HUST

Repository này chứa mã nguồn và hướng dẫn thao tác chính cho đồ án **Niryo VLA Push** trên robot Niryo Ned.

README này giúp người đọc nên chạy script nào theo thứ tự để làm lại các phần chính của dự án:

- thu dữ liệu từ robot thật,
- xử lý / chuyển đổi dữ liệu,
- train Mini-VLA,
- đánh giá Mini-VLA offline,
- chạy Mini-VLA trên robot thật,
- chạy HSV-assisted waypoint / Goal V2,
- kiểm tra Tiny-VLA adaptation,
- tham khảo baseline AI vision + MoveIt.

---

## Mục lục

1. (#1-tải-repository-về-máy)
2. (#2-cài-môi-trường-python)
3. (#3-khai-báo-robot-camera-và-đường-dẫn-làm-việc)
4. (#4-kiểm-tra-robot-và-camera)
5. (#5-thu-dữ-liệu-mini-vla)
6. (#6-chuyển-dữ-liệu-sang-định-dạng-mini-vla)
7. (#7-train-mini-vla)
8. (#8-đánh-giá-mini-vla-offline)
9. (#9-chạy-mini-vla-trên-robot-thật)
10. (#10-chạy-hsv-assisted-waypoint--goal-v2)
11. (#11-thu-dữ-liệu-tiny-vla-10d-hdf5)
12. (#12-train-tiny-vla)
13. (#13-kiểm-tra--đánh-giá-tiny-vla)
14. (#14-chạy-tiny-vla-trên-robot-thật)
15. (#15-baseline-ai-vision--moveit)
16. (#16-evidence-và-video-demo)
17. (#17-tham-khảo)

---

## 1. Tải repository về máy

Clone repository và đi vào thư mục project:

```bash
git clone https://github.com/thienden15-byte/niryo-vla-push-hust.git
cd niryo-vla-push-hust
```

Kiểm tra nhanh cấu trúc repo:

```bash
ls
```

Một số thư mục chính:

```text
scripts
models
baseline_moveit
evidence
assets
huong_dan
```

Các lệnh trong README này được chạy từ thư mục gốc của repository.

---

## 2. Cài môi trường Python

Tạo môi trường Python riêng:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Cài thư viện:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra nhanh các thư viện chính:

```bash
python3 - <<'PY'
import cv2
import torch
import pyniryo

print("OpenCV OK")
print("Torch:", torch.__version__)
print("PyNiryo OK")
PY
```

---

## 3. Khai báo robot, camera và đường dẫn làm việc

Người đọc tự đặt đường dẫn theo máy của mình.

```bash
export ROBOT_IP="<ip_robot_của_bạn>"
export CAMERA_INDEX="<id_camera_của_bạn>"

export RAW_DATA_DIR="<nơi_lưu_dữ_liệu_thô>"
export TRIM_DATA_DIR="<nơi_lưu_dữ_liệu_đã_cắt_lọc>"
export DATASET_PATH="<file_dataset_sau_khi_chuyển_đổi>"
export CHECKPOINT_PATH="<file_checkpoint_sau_khi_train>"
```

Ví dụ minh họa:

```bash
export ROBOT_IP="169.254.200.200"
export CAMERA_INDEX="3"

export RAW_DATA_DIR="./my_data/raw_push"
export TRIM_DATA_DIR="./my_data/trim_push"
export DATASET_PATH="./my_data/minivla_dataset.npz"
export CHECKPOINT_PATH="./my_checkpoints/minivla_direct.pt"
```

Ý nghĩa:

```text
ROBOT_IP        : địa chỉ IP của robot Niryo
CAMERA_INDEX    : ID camera trên máy
RAW_DATA_DIR    : thư mục lưu dữ liệu thô sau khi thu từ robot
TRIM_DATA_DIR   : thư mục lưu dữ liệu đã cắt lọc
DATASET_PATH    : file dataset sau khi chuyển sang định dạng train
CHECKPOINT_PATH : file checkpoint sau khi train model
```

---

## 4. Kiểm tra robot và camera

Kiểm tra kết nối robot:

```bash
ping "$ROBOT_IP"
```

Kiểm tra camera:

```bash
ls /dev/video*
```

Test đọc ảnh từ camera:

```bash
python3 - <<'PY'
import os
import cv2

cam = int(os.environ.get("CAMERA_INDEX", 0))
cap = cv2.VideoCapture(cam)

if not cap.isOpened():
    raise RuntimeError(f"Không mở được camera {cam}")

ret, frame = cap.read()

print("Camera:", cam)
print("Đọc ảnh:", ret)
print("Kích thước ảnh:", None if frame is None else frame.shape)

cap.release()
PY
```

Test kết nối robot bằng PyNiryo:

```bash
python3 - <<'PY'
import os
from pyniryo import NiryoRobot

ip = os.environ["ROBOT_IP"]

robot = NiryoRobot(ip)

print("Joints:", robot.get_joints())
print("Pose:", robot.get_pose())

robot.close_connection()
PY
```

Nếu robot hoặc camera chưa hoạt động, cần xử lý trước khi thu dữ liệu hoặc chạy robot thật.

---

## 5. Thu dữ liệu Mini-VLA

Mục tiêu của bước này là tạo dữ liệu thô từ robot thật.

Script chính:

```text
scripts/phase_1_data_collection/collect_data_real_v5.py
```

Dữ liệu thu được thường gồm:

```text
ảnh camera
trạng thái khớp robot
action mẫu
câu lệnh thao tác
```

Chạy thu dữ liệu:

```bash
python3 scripts/phase_1_data_collection/collect_data_real_v5.py \
  --output-dir "$RAW_DATA_DIR" \
  --instruction "push the object" \
  --camera-index "$CAMERA_INDEX"
```

Kiểm tra dữ liệu đã được tạo:

```bash
find "$RAW_DATA_DIR" -type f | head
```

Nếu dữ liệu thô đã sạch và có thể dùng trực tiếp cho bước chuyển đổi, có thể đặt:

```bash
export TRIM_DATA_DIR="$RAW_DATA_DIR"
```

Nếu dữ liệu có đoạn thừa, ví dụ đoạn robot quay về vị trí ban đầu, cần cắt lọc trước.

---

## 6. Chuyển dữ liệu sang định dạng Mini-VLA

Mục tiêu của bước này là chuyển dữ liệu episode sang một file dataset dùng để train Mini-VLA.

Script chính:

```text
scripts/phase_2_preprocessing/convert_trim_v5_to_minivla.py
```

Chạy chuyển đổi:

```bash
python3 scripts/phase_2_preprocessing/convert_trim_v5_to_minivla.py \
  --input-dir "$TRIM_DATA_DIR" \
  --output-path "$DATASET_PATH"
```

Kiểm tra file dataset đã được tạo:

```bash
ls -lh "$DATASET_PATH"
```

Sau bước này, file ở `DATASET_PATH` sẽ là input cho bước train và đánh giá Mini-VLA.

Định dạng dữ liệu Mini-VLA mong đợi:

```text
images    : ảnh RGB
states    : trạng thái robot
actions   : action mẫu
text_ids  : câu lệnh đã mã hóa
vocab     : từ điển token
```

---

## 7. Train Mini-VLA

Repository có hai hướng train Mini-VLA:

```text
Direct Mini-VLA
Diffusion Mini-VLA
```

Trong kết quả thực nghiệm của nhóm, `Direct Mini-VLA` cho kết quả offline tốt hơn, nên nên chạy hướng này trước.

Trước khi train, kiểm tra dataset:

```bash
ls -lh "$DATASET_PATH"
```

---

### 7.1 Train Direct Mini-VLA

Script chính:

```text
scripts/phase_3_training/train_minivla_direct.py
```

Chạy train bằng GPU CUDA:

```bash
python3 scripts/phase_3_training/train_minivla_direct.py \
  --dataset-path "$DATASET_PATH" \
  --save-path "$CHECKPOINT_PATH" \
  --device cuda
```

Nếu máy không có GPU CUDA, có thể chạy bằng CPU để kiểm tra code, nhưng training sẽ chậm hơn nhiều:

```bash
python3 scripts/phase_3_training/train_minivla_direct.py \
  --dataset-path "$DATASET_PATH" \
  --save-path "$CHECKPOINT_PATH" \
  --device cpu
```

Kiểm tra checkpoint sau khi train:

```bash
ls -lh "$CHECKPOINT_PATH"
```

---

### 7.2 Train Diffusion Mini-VLA

Script chính:

```text
scripts/phase_3_training/train_minivla_diffusion.py
```

Chạy train bằng GPU CUDA:

```bash
python3 scripts/phase_3_training/train_minivla_diffusion.py \
  --dataset-path "$DATASET_PATH" \
  --save-path "$CHECKPOINT_PATH" \
  --device cuda
```

Nếu máy không có GPU CUDA:

```bash
python3 scripts/phase_3_training/train_minivla_diffusion.py \
  --dataset-path "$DATASET_PATH" \
  --save-path "$CHECKPOINT_PATH" \
  --device cpu
```

Kiểm tra checkpoint sau khi train:

```bash
ls -lh "$CHECKPOINT_PATH"
```

---

## 8. Đánh giá Mini-VLA offline

Đánh giá offline dùng để kiểm tra model dự đoán action có gần với action mẫu trong dataset hay không.

Trước khi đánh giá, cần có:

```text
DATASET_PATH    : file dataset đã chuyển đổi
CHECKPOINT_PATH : checkpoint model đã train
```

Kiểm tra:

```bash
ls -lh "$DATASET_PATH"
ls -lh "$CHECKPOINT_PATH"
```

Các chỉ số thường xem:

```text
Mean L2 error        : sai số độ lớn giữa action dự đoán và action thật
Cosine similarity    : độ giống nhau về hướng giữa action dự đoán và action thật
Action norm          : độ lớn action
Zero baseline        : so với trường hợp robot không làm gì
Mean-action baseline : so với trường hợp luôn dự đoán action trung bình
```

---

### 8.1 Eval Direct Mini-VLA

Script chính:

```text
scripts/phase_4_evaluation/eval_minivla_direct.py
```

Chạy eval:

```bash
python3 scripts/phase_4_evaluation/eval_minivla_direct.py \
  --dataset-path "$DATASET_PATH" \
  --ckpt "$CHECKPOINT_PATH"
```

Nếu script eval chưa hỗ trợ truyền `dataset-path` hoặc `ckpt` từ terminal, hãy mở script và sửa đường dẫn ở phần cấu hình đầu file theo dataset/checkpoint của bạn.

---

### 8.2 Eval Diffusion Mini-VLA

Script chính:

```text
scripts/phase_4_evaluation/eval_minivla_diffusion_sameidx.py
```

Chạy eval:

```bash
python3 scripts/phase_4_evaluation/eval_minivla_diffusion_sameidx.py \
  --dataset-path "$DATASET_PATH" \
  --ckpt "$CHECKPOINT_PATH"
```

Nếu script eval chưa hỗ trợ truyền `dataset-path` hoặc `ckpt` từ terminal, hãy mở script và sửa đường dẫn ở phần cấu hình đầu file theo dataset/checkpoint của bạn.

---

## 9. Chạy Mini-VLA trên robot thật

Mục tiêu của bước này là dùng ảnh camera, câu lệnh và trạng thái hiện tại của robot để model dự đoán action, sau đó gửi action tới robot qua PyNiryo.

Script chính:

```text
scripts/phase_5_real_robot_test/run_minivla_direct_live_safe.py
```

Trước khi chạy robot thật, cần có:

```text
robot đã kết nối
camera đã kết nối
checkpoint Mini-VLA
vùng làm việc an toàn
```

Kiểm tra checkpoint:

```bash
ls -lh "$CHECKPOINT_PATH"
```

Chạy Mini-VLA với robot thật:

```bash
python3 scripts/phase_5_real_robot_test/run_minivla_direct_live_safe.py \
  --ip "$ROBOT_IP" \
  --cam "$CAMERA_INDEX" \
  --ckpt "$CHECKPOINT_PATH"
```

Nếu script có chế độ chỉ xem dự đoán hoặc dry-run, nên chạy chế độ đó trước.

Chỉ cho robot thực thi khi đã chắc chắn camera, checkpoint và vùng làm việc an toàn:

```bash
python3 scripts/phase_5_real_robot_test/run_minivla_direct_live_safe.py \
  --ip "$ROBOT_IP" \
  --cam "$CAMERA_INDEX" \
  --ckpt "$CHECKPOINT_PATH" \
  --execute
```

Trạng thái thực nghiệm hiện tại:

```text
Direct Mini-VLA có kết quả offline tốt hơn Diffusion Mini-VLA.
Khi chạy robot thật, model có thể tạo chuyển động nhưng chưa đẩy vật ổn định trong mọi trường hợp.
```

---

## 10. Chạy HSV-assisted waypoint

Pipeline này dùng HSV để hỗ trợ nhận biết vật, sau đó dùng Goal V2 để sinh waypoint khớp cho robot.

Luồng xử lý:

```text
camera image
→ HSV detection
→ object_feature [cx, cy, area]
→ Goal V2 policy
→ 15 joint waypoints
→ PyNiryo execution
```

Pipeline này ổn định hơn khi chạy robot thật, nhưng không phải VLA end-to-end hoàn toàn vì phần nhận biết vật vẫn dựa vào HSV/object feature.

---

### 10.1 Chạy HSV + Goal V2 từ camera

Script chính:

```text
scripts/phase_5_real_robot_test/hsv_goal/live_click_object_to_niryo_v2.py
```

Chạy từ camera:

```bash
python3 scripts/phase_5_real_robot_test/hsv_goal/live_click_object_to_niryo_v2.py \
  --ip "$ROBOT_IP" \
  --camera-index "$CAMERA_INDEX"
```

Nếu muốn chọn màu vật:

```bash
python3 scripts/phase_5_real_robot_test/hsv_goal/live_click_object_to_niryo_v2.py \
  --ip "$ROBOT_IP" \
  --camera-index "$CAMERA_INDEX" \
  --color green
```

Nếu script hỗ trợ dry-run, nên chạy dry-run trước khi cho robot thực thi.

---

### 10.2 Chạy Goal V2 bằng object feature nhập tay

Script chính:

```text
scripts/phase_5_real_robot_test/hsv_goal/run_goal_v2_cluster_on_niryo.py
```

Chạy với object feature tự nhập:

```bash
python3 scripts/phase_5_real_robot_test/hsv_goal/run_goal_v2_cluster_on_niryo.py \
  --ip "$ROBOT_IP" \
  --object-feature <cx> <cy> <area>
```

Trong đó:

```text
cx   : tọa độ tâm vật theo trục x sau chuẩn hóa
cy   : tọa độ tâm vật theo trục y sau chuẩn hóa
area : diện tích vật sau chuẩn hóa
```

---

## 11. Thu dữ liệu Tiny-VLA 10D HDF5

Tiny-VLA dùng định dạng dữ liệu khác Mini-VLA.

Script chính:

```text
scripts/phase_1_data_collection/tiny_vla/collect_tinyvla_10d_push_only_then_return.py
```

Mục tiêu:

Thu dữ liệu HDF5 cho Tiny-VLA, thường gồm:

```text
qpos     : trạng thái robot
action   : action 10D
image    : ảnh camera
language : câu lệnh thao tác
```

Chạy thu dữ liệu Tiny-VLA:

```bash
python3 scripts/phase_1_data_collection/tiny_vla/collect_tinyvla_10d_push_only_then_return.py \
  --robot-ip "$ROBOT_IP" \
  --camera "$CAMERA_INDEX" \
  --cam-name front \
  --out-dir "$RAW_DATA_DIR" \
  --instruction "push the green object to the right"
```

Sau khi thu, kiểm tra dữ liệu:

```bash
find "$RAW_DATA_DIR" -type f | head
```

---

## 12. Train Tiny-VLA

Tiny-VLA phụ thuộc vào repository TinyVLA gốc và môi trường train riêng.

Clone TinyVLA gốc nếu chưa có:

```bash
git clone https://github.com/liyaxuanliyaxuan/TinyVLA.git "<đường_dẫn_lưu_TinyVLA_gốc_của_bạn>"
```

Khai báo đường dẫn TinyVLA gốc:

```bash
export TINYVLA_REPO="<đường_dẫn_repo_TinyVLA_gốc_của_bạn>"
export TINYVLA_MODEL_PATH="<đường_dẫn_checkpoint_TinyVLA_sau_khi_train>"
export PYTHONPATH="$TINYVLA_REPO:$TINYVLA_REPO/llava-pythia:$PYTHONPATH"
```

Script train chính trong repository này:

```text
scripts/phase_3_training/tiny_vla/train.sh
scripts/phase_3_training/tiny_vla/train_tinyvla.py
```

Chạy train bằng shell script:

```bash
bash scripts/phase_3_training/tiny_vla/train.sh
```

Hoặc chạy bằng Python:

```bash
python3 scripts/phase_3_training/tiny_vla/train_tinyvla.py
```

Nếu cần sửa dataset path, checkpoint path hoặc model path, hãy sửa trong file train tương ứng trước khi chạy.

---

## 13. Kiểm tra / đánh giá Tiny-VLA

Các script kiểm tra Tiny-VLA nằm trong:

```text
scripts/phase_4_evaluation/tiny_vla/
```

Một số script chính:

```text
inspect_hdf5_dataset_summary.py
test_author10d_live_camera_only.py
test_infer_author10d_real_obs_dryrun.py
test_model_object_sensitivity_green.py
```

Ví dụ kiểm tra dataset HDF5:

```bash
python3 scripts/phase_4_evaluation/tiny_vla/inspect_hdf5_dataset_summary.py
```

Ví dụ kiểm tra camera live:

```bash
python3 scripts/phase_4_evaluation/tiny_vla/test_author10d_live_camera_only.py
```

Ví dụ kiểm tra inference dry-run:

```bash
python3 scripts/phase_4_evaluation/tiny_vla/test_infer_author10d_real_obs_dryrun.py
```

Các script này dùng để kiểm tra:

```text
dataset HDF5
camera live
quá trình inference
output action của model
độ nhạy của model với hình ảnh đầu vào
```

---

## 14. Chạy Tiny-VLA trên robot thật

Các script runtime Tiny-VLA nằm trong:

```text
scripts/phase_5_real_robot_test/tiny_vla/
```

Khai báo TinyVLA gốc và checkpoint:

```bash
export TINYVLA_REPO="<đường_dẫn_repo_TinyVLA_gốc_của_bạn>"
export TINYVLA_MODEL_PATH="<đường_dẫn_checkpoint_TinyVLA_sau_khi_train>"
export PYTHONPATH="$TINYVLA_REPO:$TINYVLA_REPO/llava-pythia:$PYTHONPATH"
```

Chạy Tiny-VLA rollout:

```bash
python3 scripts/phase_5_real_robot_test/tiny_vla/run_author_style_niryo_diffik_rollout.py \
  --ip "$ROBOT_IP"
```

Một số script runtime khác:

```text
scripts/phase_5_real_robot_test/tiny_vla/run_author_style_niryo_safe_rollout.py
scripts/phase_5_real_robot_test/tiny_vla/run_author_style_niryo_safe_rollout_xz_only.py
scripts/phase_5_real_robot_test/tiny_vla/run_auto_green_monotonic_push.py
scripts/phase_5_real_robot_test/tiny_vla/run_hybrid_approach_descend_push.py
```

Trạng thái thực nghiệm hiện tại:

```text
Tiny-VLA pipeline có thể load model, đọc camera, đọc robot state, inference và gửi lệnh robot.
Tuy nhiên pure Tiny-VLA chưa ổn định cho mọi vị trí vật.
```

---

## 15. Baseline AI vision + MoveIt

Baseline AI vision + MoveIt là pipeline tham khảo để kiểm tra hệ thống robot, camera, ROS 2 và MoveIt.

Baseline không phải VLA end-to-end hoàn toàn, vì quỹ đạo chuyển động chủ yếu do MoveIt lập kế hoạch.

Luồng baseline:

```text
camera / vision
→ phát hiện vật
→ gửi target
→ MoveIt lập kế hoạch IK và trajectory
→ robot / RViz / Gazebo thực thi
```

Code baseline trong repository:

```text
baseline_moveit/
baseline_moveit/src/
baseline_moveit/launch/
baseline_moveit/msg/
```

Các thành phần chính:

```text
vision node  : xử lý ảnh, phát hiện vật, gửi vị trí mục tiêu
brain node   : nhận mục tiêu, điều phối logic robot
bridge node  : kết nối giữa ROS 2 và robot
voice node   : nhận câu lệnh giọng nói nếu dùng voice
MoveIt       : lập kế hoạch chuyển động, IK, trajectory
RViz/Gazebo  : mô phỏng, kiểm tra và trực quan hóa
```

Pipeline baseline dùng để:

```text
kiểm tra camera
kiểm tra robot
kiểm tra ROS 2 / MoveIt
so sánh với các hướng học từ dữ liệu như Mini-VLA và Tiny-VLA
```

Hướng dẫn chi tiết hơn nằm trong:

```text
huong_dan/04_baseline_ai_vision_moveit.md
```

---

## 16. Evidence và video demo

Evidence nằm trong:

```text
evidence/minivla/
evidence/hsv_goal/
evidence/tinyvla/
```

Video demo nằm trong:

```text
assets/videos/
```

Một số video demo:

```text
01_baseline_moveit_demo.mp4
02_hsv_goal_v2_demo_1.mp4
03_hsv_goal_v2_demo_2.mp4
04_hsv_goal_v2_demo_3.mp4
05_hsv_goal_v2_demo_4.mp4
```

Các file evidence và video dùng để kiểm chứng kết quả đã chạy trong quá trình thực nghiệm.

---

## 17. Tham khảo

Dự án có tham khảo và mở rộng từ các hướng sau:

```text
Mini-VLA: https://github.com/keivalya/mini-vla
TinyVLA: https://github.com/liyaxuanliyaxuan/TinyVLA
```

Phần adaptation cho robot Niryo, thu dữ liệu thực nghiệm, xử lý dữ liệu, Mini-VLA, HSV-assisted waypoint / Goal V2, Tiny-VLA runtime và tổ chức repository bàn giao được thực hiện trong đồ án này.
