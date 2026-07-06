#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import speech_recognition as sr
import threading

class VoiceNode(Node):
    def __init__(self):
        super().__init__('voice_node')
        self.publisher_ = self.create_publisher(String, '/voice_command', 10)
        self.recognizer = sr.Recognizer()
        
        self.get_logger().info('🎙️ ĐÔI TAI ĐÃ BẬT. Đang lắng nghe (TIẾNG ANH)...')
        
        self.listen_thread = threading.Thread(target=self.listen_loop)
        self.listen_thread.daemon = True
        self.listen_thread.start()

    def listen_loop(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            self.get_logger().info('✅ Đã lọc ồn xong. Sẵn sàng nhận lệnh!')
            
            while rclpy.ok():
                try:
                    audio = self.recognizer.listen(source, timeout=5.0, phrase_time_limit=5.0)
                    # Chuyển lại thành Tiếng Anh
                    text = self.recognizer.recognize_google(audio, language='en-US')
                    text = text.lower()
                    
                    self.get_logger().info(f'🗣️ Bạn vừa nói: "{text}"')
                    
                    msg = String()
                    msg.data = text
                    self.publisher_.publish(msg)
                    
                except sr.WaitTimeoutError:
                    pass
                except sr.UnknownValueError:
                    self.get_logger().warn('❓ Không nghe rõ Tiếng Anh, vui lòng nói lại...')
                except sr.RequestError as e:
                    self.get_logger().error(f'🚨 Lỗi API: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = VoiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()