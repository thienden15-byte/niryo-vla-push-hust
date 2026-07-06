# Evidence

Thư mục này lưu các bằng chứng nhỏ cho đồ án Niryo VLA Push.

Không lưu dataset lớn, checkpoint lớn, raw frames hoặc video raw dài trong thư mục này.

---

## 1. Mini-VLA evidence

Thư mục:

- `evidence/minivla/logs/`
- `evidence/minivla/plots/`
- `evidence/minivla/manifests/`
- `evidence/minivla/dataset_audit/`

Nội dung chính:

- offline evaluation logs cho Diffusion Mini-VLA và Direct Mini-VLA
- biểu đồ so sánh L2 error, cosine similarity và action norm
- bảng summary offline evaluation
- SHA256 manifest
- dataset audit output cho dataset 73 episode trim

Ghi chú kết quả:

- Direct Mini-VLA cho kết quả offline tốt hơn Diffusion Mini-VLA.
- Khi chạy robot thật, Direct Mini-VLA có di chuyển robot nhưng chưa đẩy vật ổn định.

---

## 2. HSV / Goal V2 evidence

Thư mục:

- `evidence/hsv_goal/logs/`
- `evidence/hsv_goal/images/`

Nội dung chính:

- log convert/eval/infer Goal V1 và Goal V2
- ảnh camera/detection cuối khi chạy live
- ảnh locked frame và auto green detection

Ghi chú kết quả:

- HSV / Goal V2 dùng object feature `[cx, cy, area]`
- Goal V2 sinh 15 joint waypoints
- robot thật chạy ổn định hơn Direct Mini-VLA rollout
- đây không phải VLA end-to-end vì perception vẫn dựa vào HSV/object_feature

---

## 3. Tiny-VLA evidence

Thư mục:

- `evidence/tinyvla/logs/`
- `evidence/tinyvla/reports/`
- `evidence/tinyvla/summary/`

Nội dung chính:

- log compare checkpoint 2000/4000
- log rollout Tiny-VLA delta6
- log author-style 10D / DiffIK dry-run và rollout
- HDF5 dataset summary
- report image cho một số episode HDF5

Ghi chú kết quả:

- Tiny-VLA checkpoint load được
- dataset HDF5 90 episode đúng shape
- robot thật đã chạy được qua PyNiryo/DiffIK
- Pure Tiny-VLA chưa đẩy vật ổn định ở mọi vị trí
- Hybrid HSV/DiffIK chạy tốt hơn nhưng không phải VLA thuần

---

## 4. Không commit vào evidence

Không commit:

- dataset `.npz`, `.hdf5`, `.h5`
- checkpoint `.pt`, `.pth`, `.ckpt`, `.bin`, `.safetensors`
- raw frames
- video raw dài
- zip/tar dataset hoặc checkpoint
- thư mục `wandb/`, `runs/`, `__pycache__/`

Video demo ngắn được lưu riêng ở:

- `assets/videos/`
