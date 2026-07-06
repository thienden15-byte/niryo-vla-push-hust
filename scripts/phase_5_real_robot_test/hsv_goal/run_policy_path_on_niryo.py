#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import numpy as np

try:
    from pyniryo import NiryoRobot
except Exception as e:
    print("❌ Không import được pyniryo:", e)
    print("Hãy chạy trong môi trường Python mà trước đây bạn dùng để điều khiển robot.")
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True, help="IP robot Niryo, ví dụ 169.254.x.x hoặc IP bạn vẫn dùng")
    parser.add_argument("--path-file", type=str, default="goal_v1/eval/dry_run_ep026.npz")
    parser.add_argument("--which", type=str, default="main", choices=["main", "backup"])
    parser.add_argument("--velocity", type=int, default=5, help="Tốc độ robot, nên để 5 hoặc thấp hơn")
    parser.add_argument("--skip", type=int, default=1, help="Lấy mỗi n waypoint. Để 1 là chạy đầy đủ")
    parser.add_argument("--no-confirm", action="store_true")
    args = parser.parse_args()

    path_file = Path(args.path_file)
    if not path_file.exists():
        print("❌ Không thấy file path:", path_file)
        raise SystemExit(1)

    data = np.load(path_file, allow_pickle=True)

    if args.which == "main":
        path = data["main_path"].astype(np.float32)
        ok = bool(data["main_ok"])
    else:
        path = data["backup_path"].astype(np.float32)
        ok = bool(data["backup_ok"])

    if not ok:
        print("❌ Path này không pass dry-run offline. Không chạy.")
        raise SystemExit(1)

    if args.skip > 1:
        path = path[::args.skip]

    print("=" * 100)
    print("RUN POLICY PATH ON REAL NIRYO")
    print("=" * 100)
    print("Robot IP :", args.ip)
    print("Path file:", path_file)
    print("Policy   :", args.which)
    print("Velocity :", args.velocity)
    print("Waypoints:", path.shape)
    print("First q  :", np.array2string(path[0], precision=6, suppress_small=True))
    print("Last q   :", np.array2string(path[-1], precision=6, suppress_small=True))
    print("=" * 100)

    print("\n⚠️  KIỂM TRA TRƯỚC KHI CHẠY:")
    print("1. Dọn sạch vật cản quanh robot.")
    print("2. Chưa đặt vật cần đẩy vào bàn.")
    print("3. Tay để gần nút dừng khẩn cấp.")
    print("4. Nếu robot có dấu hiệu đi sai, bấm dừng ngay.")
    print("5. Đây là test chuyển động, chưa phải bản MoveIt cuối cùng.")

    if not args.no_confirm:
        ans = input("\nGõ YES để bắt đầu chạy robot thật: ").strip()
        if ans != "YES":
            print("Đã hủy.")
            return

    robot = None

    try:
        print("\nĐang kết nối robot...")
        robot = NiryoRobot(args.ip)

        print("Tắt Learning Mode...")
        robot.set_learning_mode(False)

        print(f"Đặt tốc độ robot = {args.velocity}%")
        robot.set_arm_max_velocity(args.velocity)

        print("\nDi chuyển đến điểm đầu tiên...")
        print("q_start:", np.array2string(path[0], precision=6, suppress_small=True))
        robot.move_joints(path[0].tolist())

        time.sleep(1.0)

        print("\nBắt đầu chạy trajectory...")
        list_joints = [q.tolist() for q in path]

        try:
            robot.execute_trajectory_from_poses_and_joints(
                list_joints,
                ["joint"] * len(list_joints)
            )
        except Exception as e:
            print("\n⚠️ execute_trajectory_from_poses_and_joints lỗi, thử fallback move_joints từng điểm.")
            print("Lỗi:", e)

            for i, q in enumerate(list_joints):
                print(f"Waypoint {i+1}/{len(list_joints)}")
                robot.move_joints(q)
                time.sleep(0.03)

        print("\n✅ Đã chạy xong path.")

    except KeyboardInterrupt:
        print("\n⚠️ Bị ngắt bằng Ctrl+C.")
        print("Nếu robot còn chuyển động, bấm dừng khẩn cấp trên robot.")

    except Exception as e:
        print("\n❌ Lỗi khi chạy robot:", e)
        print("Không chạy tiếp.")

    finally:
        if robot is not None:
            try:
                robot.close_connection()
            except Exception:
                pass

    print("\nDONE")


if __name__ == "__main__":
    main()
