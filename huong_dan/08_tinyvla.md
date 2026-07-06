# 08. Tiny-VLA

File này ghi lại phần Tiny-VLA trong đồ án Niryo VLA Push.

Tiny-VLA là hướng thử nghiệm mô hình VLA có sẵn, sau đó thích nghi cho robot Niryo.

Repo gốc TinyVLA:

`https://github.com/liyaxuanliyaxuan/TinyVLA`

Author root local:

`/home/thien/TinyVLA`

Niryo runtime/adaptation root:

`/home/thien/tinyvla_niryo_runtime`

---

## 1. Hai nhánh Tiny-VLA

Trong đồ án có hai nhánh Tiny-VLA cần tách rõ.

### 1.1. Delta6 branch

Nhánh này dùng dataset 73 episode trim từ Mini-VLA.

Nguồn dataset:

`/home/thien/mini-vla/mini-vla/dataset_push_real_v5_trim_manual`

Đặc điểm:

- source: 73 episode NPZ trim
- state/qpos: 6D joints
- action: 6D delta joints
- image: RGB 224x224
- instruction: `push the object`

Checkpoint:

`/home/thien/tinyvla_niryo_ckpt/train_delta6_nodeepspeed_fp32_2000steps`

`/home/thien/tinyvla_niryo_ckpt/train_delta6_nodeepspeed_fp32_4000steps`

Ghi chú:

- train command cuối cùng chưa xác nhận nguyên văn
- convert script delta6 chưa xác nhận tuyệt đối
- không ghi bừa command chưa kiểm chứng

---

### 1.2. Author-style 10D branch

Nhánh này dùng dataset HDF5 90 episode.

Dataset local:

`/home/thien/TinyVLA/data/data/niryo_push_1cam_10d_50_hdf5`

Format:

- HDF5
- 90 episode
- mỗi episode 50 bước
- `action`: `(50, 10)` float32
- `observations/qpos`: `(50, 7)` float32
- `observations/joint_positions`: `(50, 7)` float32
- `observations/qvel`: `(50, 7)` float32
- `observations/ee_pose_xyzrpy`: `(50, 6)` float32
- `observations/target_ee_pose_xyzrpy`: `(50, 6)` float32
- `observations/images/front`: `(50, 480, 640, 3)` uint8
- `language_raw`: instruction

Instruction:

`push the green object to the right`

Checkpoint:

`/home/thien/tinyvla_niryo_ckpt/author_10d_full_5000steps`

Không commit checkpoint này vì khoảng 1.1 GB.

---

## 2. Cách thu data Tiny-VLA 10D

Collector đã đưa vào repo:

`scripts/phase_1_data_collection/tiny_vla/collect_tinyvla_10d_push_only_then_return.py`

Ý tưởng thu data:

1. Bật Learning Mode.
2. Người dùng kéo tay robot để dạy đoạn đi tới vật, đẩy vật, rồi dừng.
3. Không dạy đoạn robot quay về.
4. Lưu quỹ đạo dạy tay.
5. Reset vật về vị trí đầu.
6. Robot replay quỹ đạo đã dạy.
7. Trong lúc replay, camera ghi ảnh và robot ghi trạng thái.
8. Lưu episode HDF5 50 bước.
9. Robot tự quay về điểm đầu nhưng đoạn quay về không lưu vào dataset.

Command thu data từng dùng:

`python aloha_scripts/collect_tinyvla_10d_push_only_then_return.py --robot-ip 169.254.200.200 --camera 2 --cam-name front --out-dir data/niryo_push_1cam_10d_50_hdf5 --instruction "push the green object to the right" --episode-len 50 --sample-hz 10 --replay-hz 8 --velocity 30 --return-velocity 30 --after-move-sleep 0.0 --start-settle 0.3 --width 640 --height 480 --gripper-state 0.0 --countdown 5`

Phím điều khiển:

- `s`: teach by hand, không lưu ảnh
- `p`: lưu quỹ đạo dạy
- `e`: replay push và ghi data
- `h`: quay về điểm đầu không ghi data
- `x`: bỏ trajectory
- `q`: thoát

Action 10D:

`[x, y, z, rot6d, gripper]`

Collector này lưu `action`, `language_raw`, `observations/qpos`, `observations/joint_positions`, `observations/qvel`, `observations/ee_pose_xyzrpy`, `observations/target_ee_pose_xyzrpy`, và `observations/images/front`.

---

## 3. Code đã copy vào repo

Data collection:

- `scripts/phase_1_data_collection/tiny_vla/collect_tinyvla_10d_push_only_then_return.py`

Training:

- `scripts/phase_3_training/tiny_vla/train_tinyvla.py`
- `scripts/phase_3_training/tiny_vla/train.sh`

Evaluation / inspect:

- `scripts/phase_4_evaluation/tiny_vla/export_hdf5_episode_frames.py`
- `scripts/phase_4_evaluation/tiny_vla/inspect_hdf5_dataset_summary.py`
- `scripts/phase_4_evaluation/tiny_vla/inspect_one_hdf5_episode_report_simple.py`
- `scripts/phase_4_evaluation/tiny_vla/inspect_one_hdf5_episode_split_report.py`
- `scripts/phase_4_evaluation/tiny_vla/test_author10d_live_camera_only.py`
- `scripts/phase_4_evaluation/tiny_vla/test_infer_author10d_fake_gpu.py`
- `scripts/phase_4_evaluation/tiny_vla/test_infer_author10d_real_obs_dryrun.py`
- `scripts/phase_4_evaluation/tiny_vla/dryrun_author10d_official_style_temporal.py`
- `scripts/phase_4_evaluation/tiny_vla/test_model_object_sensitivity_green.py`
- `scripts/phase_4_evaluation/tiny_vla/inspect_author10d_postprocess_compare.py`
- `scripts/phase_4_evaluation/tiny_vla/inspect_author10d_pose_conversion.py`
- `scripts/phase_4_evaluation/tiny_vla/preview_author10d_minmax_xyz_ik_guard.py`

Runtime robot thật:

- `scripts/phase_5_real_robot_test/tiny_vla/run_author_style_niryo_diffik_rollout.py`
- `scripts/phase_5_real_robot_test/tiny_vla/run_author_style_niryo_safe_rollout.py`
- `scripts/phase_5_real_robot_test/tiny_vla/run_author_style_niryo_safe_rollout_xz_only.py`
- `scripts/phase_5_real_robot_test/tiny_vla/contact_descent_diffik_assist.py`
- `scripts/phase_5_real_robot_test/tiny_vla/run_auto_green_monotonic_push.py`
- `scripts/phase_5_real_robot_test/tiny_vla/run_hybrid_approach_descend_push.py`
- `scripts/phase_5_real_robot_test/tiny_vla/run_monotonic_approach_descend_push.py`

---

## 4. Kết quả thực nghiệm

Kết quả hiện tại:

- Tiny-VLA checkpoint load được.
- Dataset 90 episode đúng shape.
- Robot thật đã chạy được qua PyNiryo/DiffIK.
- Pure Tiny-VLA chưa đẩy ổn định ở mọi vị trí.
- Model có xu hướng đi theo quỹ đạo trung bình.
- Hybrid HSV/DiffIK đẩy tốt hơn ở vị trí giữa, nhưng không phải VLA thuần.

Nhánh delta6:

- checkpoint 2000 và 4000 load được.
- output là 6D delta joints.
- checkpoint 4000 tốt hơn 2000 trong kiểm tra offline.
- robot thật chạy được pipeline, nhưng chưa có kết quả đẩy vật ổn định.

---

## 5. Không commit file lớn

Không commit:

- `/home/thien/TinyVLA/data/`
- `/home/thien/tinyvla_niryo_ckpt/`
- `/home/thien/TinyVLA/pretrained/`
- dataset `.hdf5`
- checkpoint `.bin`
- `.safetensors`
- `.pt`, `.pth`
- tar/zip dataset
- capture/video/raw frames lớn

---

## 6. Attribution

TinyVLA author repo:

`https://github.com/liyaxuanliyaxuan/TinyVLA`

Runtime/adaptation cho Niryo là phần tự viết trong:

`/home/thien/tinyvla_niryo_runtime/scripts/`
