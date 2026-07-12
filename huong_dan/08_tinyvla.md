# 08. Tiny-VLA cho robot Niryo

Tài liệu này mô tả phần thích nghi Tiny-VLA cho bài toán đẩy vật bằng robot Niryo.

Repository TinyVLA gốc:

```text
https://github.com/liyaxuanliyaxuan/TinyVLA
```

Phần thu dữ liệu Niryo, kiểm tra model, Differential IK và runtime robot thật là phần nhóm xây dựng thêm.

---

## 1. Các nhánh Tiny-VLA trong đồ án

Trong quá trình thử nghiệm có hai nhánh dữ liệu và checkpoint.

### 1.1. Nhánh Delta6

Nhánh Delta6 được chuyển đổi từ dataset Mini-VLA đã cắt bỏ pha robot quay về.

Đặc điểm:

```text
Số episode  : 73
qpos        : 6 góc khớp
action      : 6D delta joint
ảnh         : RGB
câu lệnh    : push the object
```

Nhánh này từng được train ở các mốc khoảng 2000 và 4000 bước.

Kết quả chính:

```text
- checkpoint load được;
- output là delta joint 6D;
- checkpoint 4000 bước tốt hơn checkpoint 2000 bước trong kiểm tra offline;
- chưa đạt kết quả đẩy vật ổn định trên robot thật.
```

Nhánh Delta6 được giữ lại để tham khảo lịch sử thử nghiệm, không phải nhánh runtime chính hiện tại.

---

### 1.2. Nhánh author-style 10D

Đây là nhánh Tiny-VLA chính hiện tại.

Dataset có cấu trúc HDF5 gần với định dạng của tác giả:

```text
Số episode       : 90
Số bước/episode  : 50
Tần số lấy mẫu   : 10 Hz
Camera            : front
Kích thước ảnh   : 480 × 640
qpos              : 7D
action            : 10D
câu lệnh          : push the green object to the right
```

qpos 7D gồm:

```text
6 góc khớp Niryo + 1 trạng thái gripper
```

action 10D gồm:

```text
XYZ 3D + rotation 6D + gripper 1D
```

Dạng tổng quát:

```text
[x, y, z,
 rot6d_1, rot6d_2, rot6d_3,
 rot6d_4, rot6d_5, rot6d_6,
 gripper]
```

Các dataset chính trong một episode:

```text
action
language_raw
observations/qpos
observations/joint_positions
observations/qvel
observations/ee_pose_xyzrpy
observations/target_ee_pose_xyzrpy
observations/images/front
```

Checkpoint chính đã dùng:

```text
author_10d_full_5000steps
```

Backbone:

```text
LLaVA-Pythia-400M
```

Dataset, checkpoint và backbone không được lưu trong GitHub do dung lượng lớn.

---

## 2. Khai báo đường dẫn

Không sửa trực tiếp đường dẫn máy cá nhân trong source code.

Khai báo các biến môi trường:

```bash
export ROBOT_IP="<địa_chỉ_IP_robot>"
export CAMERA_INDEX="<camera_index>"

export TINYVLA_REPO="<đường_dẫn_repository_TinyVLA_gốc>"
export TINYVLA_DATASET_DIR="<đường_dẫn_dataset_HDF5>"
export TINYVLA_MODEL_BASE="<đường_dẫn_LLaVA-Pythia-400M>"
export TINYVLA_MODEL_PATH="<đường_dẫn_checkpoint_TinyVLA>"
export TINYVLA_OUTPUT_DIR="<thư_mục_lưu_kết_quả_debug>"
```

Thêm TinyVLA vào Python path:

```bash
export PYTHONPATH="$TINYVLA_REPO:$TINYVLA_REPO/llava-pythia:$PYTHONPATH"
```

Các script vẫn có giá trị mặc định để tương thích với máy đã dùng trong quá trình thực nghiệm. Khi chạy trên máy khác, nên khai báo đầy đủ các biến môi trường trên.

---

## 3. Thu dữ liệu author-style 10D

Collector:

```text
scripts/phase_1_data_collection/tiny_vla/collect_tinyvla_10d_push_only_then_return.py
```

Quy trình thu dữ liệu:

```text
1. Bật Learning Mode.
2. Người dùng kéo tay robot để dạy quỹ đạo tiếp cận và đẩy vật.
3. Chỉ dạy pha đẩy, không dạy pha quay về.
4. Lưu quỹ đạo dạy tay.
5. Đặt lại vật về vị trí ban đầu.
6. Robot replay quỹ đạo.
7. Trong lúc replay, camera và robot state được ghi lại.
8. Episode được resample thành 50 bước.
9. Robot quay về điểm đầu nhưng pha quay về không được lưu vào dataset.
```

Lệnh chạy:

```bash
python3 scripts/phase_1_data_collection/tiny_vla/collect_tinyvla_10d_push_only_then_return.py \
  --robot-ip "$ROBOT_IP" \
  --camera "$CAMERA_INDEX" \
  --cam-name front \
  --out-dir "$TINYVLA_DATASET_DIR" \
  --instruction "push the green object to the right" \
  --episode-len 50 \
  --sample-hz 10 \
  --replay-hz 8 \
  --velocity 30 \
  --return-velocity 30 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --countdown 5 \
  --start-settle 0.3 \
  --after-move-sleep 0.0 \
  --gripper-state 0.0
```

Phím điều khiển của collector:

```text
s : bắt đầu dạy tay
p : lưu quỹ đạo đã dạy
e : replay và ghi episode
h : đưa robot quay về điểm đầu
x : bỏ quỹ đạo hiện tại
q : thoát
```

Có thể tắt tự động quay về bằng:

```bash
--no-auto-return
```

Kiểm tra số episode:

```bash
find "$TINYVLA_DATASET_DIR" \
  -maxdepth 1 \
  -name 'episode_*.hdf5' \
  | wc -l
```

---

## 4. Train Tiny-VLA

Các file train trong repository:

```text
scripts/phase_3_training/tiny_vla/train.sh
scripts/phase_3_training/tiny_vla/train_tinyvla.py
```

`train_tinyvla.py` đọc cấu hình dataset từ:

```python
TASK_CONFIGS[task_name]
```

Vì vậy cần khai báo task Niryo trong TinyVLA gốc, thường tại:

```text
$TINYVLA_REPO/aloha_scripts/constants.py
```

Tên task đã dùng:

```text
niryo_push_1cam_10d_50_author
```

Task cần chứa các thông tin chính:

```text
dataset_dir  : đường dẫn dataset HDF5
camera_names : ["front"]
num_episodes : 90
episode_len  : 50
```

File `train.sh` hiện là template tham khảo. Trước khi chạy phải kiểm tra và thay các giá trị:

```text
/path/to/save_dir
/path/to/pretrained_vlm
task_name
số GPU
max_steps
save_steps
```

Không chạy nguyên bản `train.sh` nếu vẫn còn đường dẫn `/path/to/...`.

Cấu hình chính đã dùng trong thử nghiệm:

```text
Backbone         : LLaVA-Pythia-400M
Fine-tuning      : LoRA
Action decoder   : diffusion
Action dimension : 10
Action chunk     : 16 bước
Số bước train    : khoảng 5000
```

Checkpoint thường chứa:

```text
dataset_stats.pkl
adapter_config.json
adapter_model.bin
non_lora_trainables.bin
config.json
trainer_state.json
```

---

## 5. Kiểm tra dataset và model

Các script đánh giá nằm trong:

```text
scripts/phase_4_evaluation/tiny_vla/
```

### 5.1. Kiểm tra cấu trúc HDF5

```bash
python3 scripts/phase_4_evaluation/tiny_vla/inspect_hdf5_dataset_summary.py
```

Các script inspect khác:

```text
export_hdf5_episode_frames.py
inspect_one_hdf5_episode_report_simple.py
inspect_one_hdf5_episode_split_report.py
```

### 5.2. Kiểm tra quy ước rotation 6D

```bash
python3 scripts/phase_4_evaluation/tiny_vla/check_rot6d_rpy_convention.py \
  --root "$TINYVLA_DATASET_DIR" \
  --num-files 10
```

Quy ước rot6D đã xác nhận:

```text
[R00, R01, R10, R11, R20, R21]
```

Đây là hai cột đầu của ma trận quay được flatten theo cách xen kẽ.

### 5.3. Kiểm tra camera

```bash
python3 scripts/phase_4_evaluation/tiny_vla/test_author10d_live_camera_only.py
```

### 5.4. Inference dry-run

```bash
python3 scripts/phase_4_evaluation/tiny_vla/test_infer_author10d_real_obs_dryrun.py \
  --ip "$ROBOT_IP" \
  --instruction "push the green object to the right"
```

Dry-run đọc ảnh, qpos và chạy model nhưng không gửi lệnh chuyển động tới robot.

### 5.5. Kiểm tra độ nhạy theo ảnh

```bash
python3 scripts/phase_4_evaluation/tiny_vla/test_author10d_image_sensitivity_manual.py \
  --ip "$ROBOT_IP" \
  --instruction "push the green object to the right" \
  --num-scenes 4
```

Script dùng nhiều cảnh ảnh khác nhau để kiểm tra output của model có thay đổi theo vị trí vật hay không.

### 5.6. Kiểm tra min/max và action 10D

```bash
python3 scripts/phase_4_evaluation/tiny_vla/inspect_author10d_rot6d_minmax_interleaved.py \
  --ip "$ROBOT_IP"
```

Các script kiểm tra hậu xử lý khác:

```text
inspect_author10d_postprocess_compare.py
inspect_author10d_pose_conversion.py
dryrun_author10d_official_style_temporal.py
test_model_object_sensitivity_green.py
test_infer_author10d_fake_gpu.py
```

### 5.7. Kiểm tra IK guard

```bash
python3 scripts/phase_4_evaluation/tiny_vla/preview_author10d_minmax_xyz_ik_guard.py \
  --ip "$ROBOT_IP" \
  --samples 5 \
  --max-step 0.018 \
  --min-z 0.055 \
  --max-x 0.320 \
  --max-abs-y 0.180
```

Script dùng để kiểm tra giới hạn Cartesian, delta joint và khả năng giải IK trước khi cho robot chuyển động.

---

## 6. Helper chung cho runtime

Helper đã được đưa vào repository:

```text
scripts/common/tiny_vla/run_author10d_fixed50_xyz_chunk_live.py
```

Các script đánh giá và runtime tải helper bằng đường dẫn tương đối từ repository, không còn phụ thuộc vào thư mục runtime ngoài GitHub.

Helper thực hiện các chức năng chính:

```text
load model và tokenizer
đọc dataset_stats.pkl
chuẩn bị ảnh và qpos
chạy Tiny-VLA inference
giải chuẩn hóa action bằng min/max
xử lý action chunk
```

---

## 7. Chạy robot thật

Các script runtime nằm trong:

```text
scripts/phase_5_real_robot_test/tiny_vla/
```

Runtime chính hiện tại:

```text
run_author_style_niryo_diffik_chunk_blocks.py
```

Các runtime quan trọng khác:

```text
run_author_style_niryo_diffik_rollout.py
run_author_style_niryo_diffik_chunk_manual.py
run_author_style_niryo_safe_rollout.py
run_author_style_niryo_safe_rollout_xz_only.py
contact_descent_diffik_assist.py
run_auto_green_monotonic_push.py
run_hybrid_approach_descend_push.py
run_monotonic_approach_descend_push.py
```

### 7.1. Pipeline runtime

```text
Ảnh camera
+ câu lệnh
+ qpos 7D hiện tại
        ↓
LLaVA-Pythia + LoRA
        ↓
Diffusion action decoder
        ↓
Action chunk 16 × 10D
        ↓
Giải chuẩn hóa min/max
        ↓
Lấy XYZ
        ↓
Differential IK
        ↓
Giới hạn Cartesian và delta joint
        ↓
PyNiryo move_joints()
```

Model dự đoán đầy đủ:

```text
XYZ + rot6D + gripper
```

Runtime robot thật hiện chỉ dùng phần XYZ.

rot6D và gripper chưa được gửi trực tiếp tới robot nhằm giảm rủi ro khi policy chưa đủ ổn định.

---

## 8. Differential IK

Differential IK là lớp chuyển đổi và an toàn do nhóm xây dựng thêm, không phải thành phần gốc của TinyVLA.

Các hàm chính:

```text
numerical_xyz_jacobian
solve_diffik
move_joints_compat
```

Vai trò:

```text
- chuyển delta XYZ thành delta joint;
- giới hạn độ lớn bước Cartesian;
- giới hạn delta joint;
- kiểm tra giới hạn workspace;
- tương thích các phiên bản PyNiryo;
- giảm nguy cơ robot chuyển động đột ngột.
```

---

## 9. Dry-run runtime chính

Chạy không có `--execute`:

```bash
python3 \
  scripts/phase_5_real_robot_test/tiny_vla/run_author_style_niryo_diffik_chunk_blocks.py \
  --ip "$ROBOT_IP" \
  --instruction "push the green object to the right" \
  --blocks 4 \
  --chunk-steps 16 \
  --max-cart-step 0.020 \
  --max-dq 0.055 \
  --min-z 0.080 \
  --max-x 0.360 \
  --max-abs-y 0.220 \
  --velocity 6 \
  --sleep 0.18
```

Không có `--execute` thì script không gửi lệnh di chuyển tới robot.

---

## 10. Thực thi runtime chính

Chỉ chạy sau khi dry-run không phát hiện bất thường:

```bash
python3 \
  scripts/phase_5_real_robot_test/tiny_vla/run_author_style_niryo_diffik_chunk_blocks.py \
  --ip "$ROBOT_IP" \
  --instruction "push the green object to the right" \
  --blocks 4 \
  --chunk-steps 16 \
  --max-cart-step 0.020 \
  --max-dq 0.055 \
  --min-z 0.080 \
  --max-x 0.360 \
  --max-abs-y 0.220 \
  --velocity 6 \
  --sleep 0.18 \
  --execute \
  --confirm YES_CHUNK_BLOCKS
```

Quy tắc an toàn:

```text
- luôn đặt tay gần nút dừng khẩn cấp;
- kiểm tra workspace trước khi chạy;
- không để người đứng trong vùng hoạt động;
- kiểm tra giới hạn X, Y và Z;
- chạy tốc độ thấp trong lần đầu;
- dừng ngay khi robot chuyển động sai hướng;
- không bỏ qua bước dry-run.
```

---

## 11. Kết quả thực nghiệm

Pipeline author-style 10D đã thực hiện được đầy đủ:

```text
đọc camera
→ đọc qpos
→ load checkpoint
→ inference action chunk 16 × 10D
→ giải chuẩn hóa action
→ lấy XYZ
→ Differential IK
→ move_joints()
```

Trong một số lần thử, robot đã:

```text
- tiếp cận vật;
- chạm vật;
- đẩy được vật ở một số action đầu của chunk.
```

Hạn chế hiện tại:

```text
- kết quả chưa ổn định ở mọi vị trí vật;
- khả năng grounding theo ảnh còn hạn chế;
- output có xu hướng gần với quỹ đạo trung bình trong dataset;
- rot6D và gripper chưa được thực thi trực tiếp;
- pure Tiny-VLA chưa đạt tỷ lệ thành công ổn định.
```

Kết quả hiện tại được xem là kiểm chứng thành công pipeline Tiny-VLA trên robot thật, chưa phải một policy đẩy vật hoàn toàn ổn định.

Các hướng hybrid HSV/Differential IK có thể đẩy ổn định hơn ở một số vị trí, nhưng không được coi là Tiny-VLA thuần.

---

## 12. Các file không đưa lên GitHub

Không commit:

```text
dataset HDF5
ảnh thô và raw frame
checkpoint model
backbone pretrained
dataset_stats của checkpoint lớn
file .bin
file .safetensors
file .pt và .pth
file tar hoặc zip dataset
video dung lượng lớn
thư mục outputs/
```

Các loại đường dẫn cần giữ ngoài repository:

```text
TINYVLA_DATASET_DIR
TINYVLA_MODEL_PATH
TINYVLA_MODEL_BASE
```

---

## 13. Attribution

TinyVLA gốc:

```text
https://github.com/liyaxuanliyaxuan/TinyVLA
```

Các thành phần được nhóm bổ sung cho Niryo:

```text
- collector author-style HDF5 10D;
- chuyển đổi pose và rotation 6D;
- dry-run và inspection scripts;
- image-sensitivity tests;
- min/max action post-processing;
- Differential IK;
- workspace và joint safety guards;
- chunk-manual và chunk-block runtime;
- tích hợp PyNiryo để chạy robot thật.
```
