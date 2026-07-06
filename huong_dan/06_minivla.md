# 06. Mini-VLA

File này ghi lại phần Mini-VLA trong đồ án Niryo VLA Push.

Mini-VLA là hướng học hành động chính trong đồ án. Mô hình nhận ảnh, câu lệnh và trạng thái robot, sau đó dự đoán action dạng 6D delta joints.

---

## 1. Vai trò của Mini-VLA

Mini-VLA được dùng để kiểm tra hướng Vision-Language-Action nhẹ cho robot Niryo Ned.

Đầu vào của mô hình:

- ảnh RGB 224x224
- text instruction đã tokenize
- trạng thái robot 6D joints

Đầu ra của mô hình:

- 6D delta joints

Trong đồ án có hai biến thể Mini-VLA:

- Diffusion Mini-VLA
- Direct Mini-VLA

Direct Mini-VLA được ưu tiên thử nghiệm trên robot thật vì kết quả offline tốt hơn Diffusion Mini-VLA.

---

## 2. Dataset dùng cho Mini-VLA

Dataset converted dùng để train Mini-VLA:

`~/mini-vla/mini-vla/data/niryo_push_real_v5_trim_minivla.npz`

Dataset này được tạo từ dataset sạch:

`~/mini-vla/mini-vla/dataset_push_real_v5_trim_manual`

Thông tin chính:

- 73 episode gốc
- 3991 sample sau khi convert
- images: `(3991, 224, 224, 3)`, dtype `uint8`
- states: `(3991, 6)`, dtype `float32`
- actions: `(3991, 6)`, dtype `float32`
- text_ids: `(3991, 16)`, dtype `int64`

Vocabulary:

- `<pad>`: 0
- `<unk>`: 1
- `push`: 2
- `the`: 3
- `object`: 4

Instruction chính:

`push the object`

Không commit file dataset `.npz` lên GitHub vì file converted khoảng 377 MB.

---

## 3. Code Mini-VLA trong repo này

Model files:

- `models/mini_vla/encoders.py`
- `models/mini_vla/fusion.py`
- `models/mini_vla/diffusion_head.py`
- `models/mini_vla/vla_diffusion_policy.py`
- `models/mini_vla/vla_direct_policy.py`

Training scripts:

- `scripts/phase_3_training/train_minivla_diffusion.py`
- `scripts/phase_3_training/train_minivla_direct.py`

Evaluation scripts:

- `scripts/phase_4_evaluation/eval_minivla_diffusion_sameidx.py`
- `scripts/phase_4_evaluation/eval_minivla_direct.py`

Real robot rollout script:

- `scripts/phase_5_real_robot_test/run_minivla_direct_live_safe.py`

Đường dẫn gốc trên máy:

`~/mini-vla/mini-vla`

---

## 4. Train Diffusion Mini-VLA

Script gốc:

`~/mini-vla/mini-vla/scripts/train.py`

Script trong repo này:

`scripts/phase_3_training/train_minivla_diffusion.py`

Command train đã dùng cho bản V5 trim:

`python -m scripts.train --dataset-path data/niryo_push_real_v5_trim_minivla.npz --resize-to 224 --batch-size 32 --epochs 80 --lr 1e-4 --d-model 128 --diffusion-T 50 --save-path /content/drive/MyDrive/minivla_push_trim_v5/minivla_push_trim_v5.pt --device cuda`

Checkpoint Diffusion local:

`~/mini-vla/mini-vla/checkpoints/minivla_push_trim_v5.pt`

Không commit checkpoint `.pt` trực tiếp lên GitHub.

---

## 5. Train Direct Mini-VLA

Script gốc:

`~/mini-vla/mini-vla/scripts/train_direct.py`

Script trong repo này:

`scripts/phase_3_training/train_minivla_direct.py`

Command train đã dùng:

`python -m scripts.train_direct --dataset-path data/niryo_push_real_v5_trim_minivla.npz --resize-to 224 --batch-size 64 --epochs 120 --lr 1e-4 --d-model 128 --save-path /content/drive/MyDrive/minivla_push_trim_v5/minivla_push_trim_v5_direct.pt --device cuda`

Checkpoint Direct local:

`~/mini-vla/mini-vla/checkpoints/minivla_push_trim_v5_direct.pt`

Không commit checkpoint `.pt` trực tiếp lên GitHub.

---

## 6. Eval offline Diffusion Mini-VLA

Script gốc:

`~/mini-vla/mini-vla/eval_diffusion_local_sameidx.py`

Script trong repo này:

`scripts/phase_4_evaluation/eval_minivla_diffusion_sameidx.py`

Command eval:

`cd ~/mini-vla/mini-vla`

`python3 eval_diffusion_local_sameidx.py | tee evidence_minivla_v5/logs/diffusion_offline_eval_300_sameidx_rerun.txt`

Kết quả chính:

- mean L2: 0.0622642897
- zero baseline: 0.0310210828
- mean-action baseline: 0.0236045495
- cosine mean: 0.1852063835
- pred norm: 0.0606883690
- true norm: 0.0310210828

Nhận xét:

Diffusion Mini-VLA có train được nhưng kết quả offline kém hơn zero baseline và mean-action baseline.

---

## 7. Eval offline Direct Mini-VLA

Script gốc:

`~/mini-vla/mini-vla/eval_direct_local.py`

Script trong repo này:

`scripts/phase_4_evaluation/eval_minivla_direct.py`

Command eval:

`cd ~/mini-vla/mini-vla`

`python3 eval_direct_local.py | tee evidence_minivla_v5/logs/direct_offline_eval_300_rerun.txt`

Kết quả chính:

- mean L2: 0.0133618861
- zero baseline: 0.0310210828
- mean-action baseline: 0.0236045495
- cosine mean: 0.8578314185
- pred norm: 0.0279723424
- true norm: 0.0310210828

Nhận xét:

Direct Mini-VLA có kết quả offline tốt hơn rõ rệt so với Diffusion Mini-VLA và tốt hơn các baseline đơn giản.

---

## 8. Rollout Direct Mini-VLA trên robot thật

Script gốc:

`~/mini-vla/mini-vla/run_minivla_direct_live_safe.py`

Script trong repo này:

`scripts/phase_5_real_robot_test/run_minivla_direct_live_safe.py`

Checkpoint Direct dùng cho rollout:

`checkpoints/minivla_push_trim_v5_direct.pt`

Default hiện tại trong script:

- IP robot: `169.254.200.200`
- camera index: `3`
- steps: `1`
- velocity: `4`
- action scale: `0.25`
- max delta: `0.008`
- execute: `False` nếu không thêm `--execute`

Ghi chú:

Có ghi nhận từng thử nghiệm rollout với cấu hình mạnh hơn như steps 40, velocity 5, action-scale 0.45, max-delta 0.012 và có `--execute`, nhưng audit hiện tại chưa tìm thấy log command cuối cùng đầy đủ. Vì vậy không ghi nó là command chính thức nếu chưa có bằng chứng log.

---

## 9. Kết quả robot thật

Theo ghi nhận thử nghiệm:

- pipeline inference và control chạy được
- robot thật có di chuyển
- không có lỗi lệnh nghiêm trọng trong quá trình chạy
- robot chưa tiếp cận vật đủ gần
- chưa chạm hoặc đẩy vật ổn định
- task push chưa thành công ổn định

Kết luận thận trọng:

Direct Mini-VLA học được xu hướng action tốt trong offline evaluation, nhưng khi triển khai robot thật vẫn chưa đủ ổn định để đẩy vật thành công.

---

## 10. Evidence Mini-VLA

Các file evidence gốc nằm tại:

`~/mini-vla/mini-vla/evidence_minivla_v5`

Các file quan trọng:

- `logs/direct_offline_eval_300_rerun.txt`
- `logs/diffusion_offline_eval_300_sameidx_rerun.txt`
- `manifests/direct_evidence_sha256.txt`
- `manifests/diffusion_evidence_sha256.txt`
- `plots/minivla_l2_error_comparison.png`
- `plots/minivla_cosine_comparison.png`
- `plots/minivla_action_norm_comparison.png`
- `plots/minivla_offline_eval_table.png`

Trong repo bàn giao, evidence nhỏ có thể copy sang `evidence/`. Dataset và checkpoint lớn thì không commit trực tiếp.

---

## 11. Không commit các file lớn

Không commit:

- `data/*.npz`
- `dataset_push_real_v5_trim_manual/`
- `dataset_push_real_v5_trim_manual.zip`
- `colab_minivla_push_trim_v5.zip`
- `checkpoints/*.pt`
- `checkpoints/*.pth`
- `checkpoints/*.ckpt`
- `raw_frames/`
- `wandb/`
- `runs/`
- `__pycache__/`

Dataset và checkpoint nên lưu bằng Google Drive, GitHub Release hoặc nền tảng lưu trữ ngoài.

---

## 12. Ghi chú cho người phát triển sau

Mini-VLA hiện tại chủ yếu chứng minh được pipeline học hành động và đánh giá offline.

Hạn chế chính:

- dataset còn nhỏ
- instruction chỉ có một câu
- object distribution chưa đủ đa dạng
- model có thể học quỹ đạo trung bình
- rollout robot thật dễ tích lũy sai số
- grounding thị giác chưa đủ mạnh

Hướng cải thiện:

- tăng số episode
- thêm nhiều vị trí vật thể
- thêm dữ liệu correction
- tăng đa dạng camera/object
- thử image encoder mạnh hơn
- thêm closed-loop correction khi robot đi lệch
