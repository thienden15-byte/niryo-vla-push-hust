#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import threading 

from my_robot_interfaces.msg import ObjectPose
from std_msgs.msg import String 

class MiniVLAModel(nn.Module):
    def __init__(self, num_commands=6):
        super(MiniVLAModel, self).__init__()
        resnet = models.resnet18(weights=None)
        self.vision_frozen = nn.Sequential(*list(resnet.children())[:6]) 
        self.vision_tunable = nn.Sequential(*list(resnet.children())[6:8])
        self.language = nn.Embedding(num_commands, 128)
        self.decoder = nn.Sequential(
            nn.Conv2d(640, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, image, cmd_id):
        x = self.vision_frozen(image)
        img_feat = self.vision_tunable(x)
        lang_feat = self.language(cmd_id).unsqueeze(2).unsqueeze(3).expand(-1, -1, 7, 7)
        fused = torch.cat([img_feat, lang_feat], dim=1)
        return self.decoder(fused)

class VLAVisionNodeV2(Node):
    def __init__(self):
        super().__init__('vla_vision_node_v2')
        
        self.publisher_ = self.create_publisher(ObjectPose, '/detected_object', 10)
        self.voice_sub = self.create_subscription(String, '/voice_command', self.voice_callback, 10)
        
        self.IMG_SIZE = 600
        self.ALPHA = 0.2
        
        self.smooth_x = None
        self.smooth_y = None
        self.x_ratio = 0.0  
        self.y_ratio = 0.0  
        self.clicks = []
        self.transform_matrix = None
        
        self.current_cmd_id = 0
        self.current_cmd_text = "pick up the red circle"
        self.HOTKEYS = {
            ord('1'): ("pick up the red circle", 0),
            ord('2'): ("pick up the red square", 1),
            ord('3'): ("pick up the green circle", 2),
            ord('4'): ("pick up the green square", 3),
            ord('5'): ("pick up the blue circle", 4),
            ord('6'): ("pick up the blue square", 5)
        }
        
        # TỪ ĐIỂN TIẾNG ANH ĐỂ NHẬN DIỆN TỪ KHÓA
        self.ENGLISH_DICT = {
            "red circle": ("pick up the red circle", 0),
            "red square": ("pick up the red square", 1),
            "green circle": ("pick up the green circle", 2),
            "green square": ("pick up the green square", 3),
            "blue circle": ("pick up the blue circle", 4),
            "blue square": ("pick up the blue square", 5)
        }
        
        self.get_logger().info("🧠 Đang nạp bộ não VLA...")
        self.device = torch.device("cpu")
        self.model = MiniVLAModel().to(self.device)
        base_path = os.path.expanduser('~/niryo_ws/src/my_robot_control/my_robot_control/dataset')
        brain_path = os.path.join(base_path, 'robot_brain_master_v14.pth')
        self.model.load_state_dict(torch.load(brain_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # -------------------------------------------------------------------------
        # ĐOẠN CODE TỰ ĐỘNG DÒ TÌM CỔNG CAMERA IMX179 (KHÔNG LO ĐỔI SỐ)
        # -------------------------------------------------------------------------
        cam_idx = 3  # Giá trị mặc định phòng hờ
        
        # Quét qua các cổng video từ 0 đến 9 trong hệ thống Linux
        for i in range(10):
            name_path = f"/sys/class/video4linux/video{i}/name"
            if os.path.exists(name_path):
                try:
                    with open(name_path, "r") as f:
                        camera_device_name = f.read().strip()
                    # Kiểm tra xem tên thiết bị có đúng là con IMX179 (USB Camera2) không
                    if "USB Camera2" in camera_device_name:
                        cam_idx = i
                        break  # Tìm thấy cổng nhỏ nhất là dừng lại luôn
                except Exception:
                    pass

        self.get_logger().info(f"🔍 [MẮT THẦN AI] Đã tự động tìm thấy IMX179 tại cổng: /dev/video{cam_idx}")
        self.get_logger().info("⚙️ Đang nạp cấu hình phần cứng tối ưu...")

        # Ép xung cấu hình theo biến số cổng cam_idx vừa tìm được động
        os.system(f"v4l2-ctl -d /dev/video{cam_idx} -c auto_exposure=3 > /dev/null 2>&1")
        os.system(f"v4l2-ctl -d /dev/video{cam_idx} -c brightness=0 > /dev/null 2>&1")
        os.system(f"v4l2-ctl -d /dev/video{cam_idx} -c gain=50 > /dev/null 2>&1")
        os.system(f"v4l2-ctl -d /dev/video{cam_idx} -c gamma=300 > /dev/null 2>&1")
        os.system(f"v4l2-ctl -d /dev/video{cam_idx} -c contrast=40 > /dev/null 2>&1")
        os.system(f"v4l2-ctl -d /dev/video{cam_idx} -c sharpness=45 > /dev/null 2>&1")

        # Khởi tạo OpenCV với biến số cổng vừa quét được
        self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        
        # Vũ khí chống delay, ép lấy khung hình thời gian thực
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Ép chuẩn MJPEG và phân giải 1280x720 để chạy mượt 30 FPS
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            self.get_logger().error(f"❌ Lỗi: Không thể kết nối Webcam IMX179 tại /dev/video{cam_idx}! Kiểm tra lại dây cáp.")
            return
        # -------------------------------------------------------------------------
            
        cv2.namedWindow("Live Camera")
        cv2.setMouseCallback("Live Camera", self.get_markers)
        
        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info("👉 BƯỚC 1: Click 4 góc của sa bàn!")

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def get_markers(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.clicks) < 4:
            self.clicks.append((x, y))

    # CƠ CHẾ XỬ LÝ LỆNH TIẾNG ANH & TỰ ĐỘNG GẮP
    def voice_callback(self, msg):
        en_text = msg.data.lower()
        self.get_logger().info(f'👂 Mắt thần vừa nghe được lệnh: "{en_text}"')
        
        matched = False
        for key, (cmd_text, cmd_id) in self.ENGLISH_DICT.items():
            if key in en_text: 
                self.current_cmd_text = cmd_text
                self.current_cmd_id = cmd_id
                self.get_logger().info(f'🔄 Đã khóa mục tiêu: "{cmd_text}"')
                matched = True
                break
                
        if matched:
            if self.transform_matrix is not None:
                self.get_logger().info('⏳ AI đang dời tâm ngắm... (đợi 1 giây)')
                threading.Timer(1.0, self.trigger_auto_pick).start()
            else:
                self.get_logger().warn('⚠️ Khóa sa bàn (click 4 góc) trước khi ra lệnh bằng giọng nói!')
        else:
            self.get_logger().warn('🤷 Không tìm thấy lệnh màu sắc nào trong câu nói.')

    def trigger_auto_pick(self):
        self.get_logger().info(f"🚀 [VOICE CONTROL] TỰ ĐỘNG PHÁT LỆNH GẮP: Ratio_X={self.x_ratio:.3f}, Ratio_Y={self.y_ratio:.3f}")
        msg = ObjectPose()
        msg.x = float(self.x_ratio)  
        msg.y = float(self.y_ratio)  
        msg.z = 0.0             
        msg.object_name = self.current_cmd_text
        self.publisher_.publish(msg)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        if self.transform_matrix is None:
            display_frame = frame.copy()
            for pt in self.clicks:
                cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)
            cv2.putText(display_frame, f"CLICK 4 GOC: {len(self.clicks)}/4", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Live Camera", display_frame)
            
            if len(self.clicks) == 4:
                pts = np.array(self.clicks, dtype="float32")
                ordered_pts = self.order_points(pts)
                dst_pts = np.array([[0, 0], [self.IMG_SIZE - 1, 0], [self.IMG_SIZE - 1, self.IMG_SIZE - 1], [0, self.IMG_SIZE - 1]], dtype="float32")
                self.transform_matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
                self.get_logger().info("✅ Đã khóa sa bàn! Có thể nói Tiếng Anh ngay lúc này.")
                cv2.destroyWindow("Live Camera")
        else:
            warped_img = cv2.warpPerspective(frame, self.transform_matrix, (self.IMG_SIZE, self.IMG_SIZE))
            
            pil_img = Image.fromarray(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
            tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
            cmd_tensor = torch.tensor([self.current_cmd_id], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                heatmap_pred = self.model(tensor_img, cmd_tensor)
            
            hm_numpy = heatmap_pred.squeeze().cpu().numpy()
            y_max, x_max = np.unravel_index(np.argmax(hm_numpy), hm_numpy.shape)
            
            pred_x = (x_max / 56.0) * self.IMG_SIZE
            pred_y = (y_max / 56.0) * self.IMG_SIZE
            
            if self.smooth_x is None or self.smooth_y is None:
                self.smooth_x = pred_x
                self.smooth_y = pred_y
            else:
                self.smooth_x = (self.ALPHA * pred_x) + ((1 - self.ALPHA) * self.smooth_x)
                self.smooth_y = (self.ALPHA * pred_y) + ((1 - self.ALPHA) * self.smooth_y)
            
            self.x_ratio = (self.IMG_SIZE - self.smooth_y) / self.IMG_SIZE
            self.y_ratio = ((self.IMG_SIZE / 2) - self.smooth_x) / self.IMG_SIZE
            
            cv2.drawMarker(warped_img, (int(self.smooth_x), int(self.smooth_y)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(warped_img, f"Cmd: {self.current_cmd_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("AI Live View", warped_img)

        # Phím dự phòng
        key = cv2.waitKey(1) & 0xFF
        if key in self.HOTKEYS:
            self.current_cmd_text, self.current_cmd_id = self.HOTKEYS[key]
            self.get_logger().info(f"Đổi lệnh thủ công: {self.current_cmd_text}")
        elif key == ord(' '): 
            if self.transform_matrix is not None:
                self.trigger_auto_pick()

def main(args=None):
    rclpy.init(args=args)
    node = VLAVisionNodeV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()