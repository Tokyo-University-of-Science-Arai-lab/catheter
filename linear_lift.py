# target_publisher_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String

class TargetPublisher(Node):
    def __init__(self):
        super().__init__('target_publisher')

        self.publisher_ = self.create_publisher(Float32, '/target_mm', 10)
        self.subscription = self.create_subscription(String, '/input', self.callback, 10)

        self.get_logger().info('TargetPublisher node started')

    def callback(self, msg: String):
        # 既存仕様（/input 受けたら 300 を出す）
        self.publish_target_mm(300.0)

    def publish_target_mm(self, value_mm: float):
        target_msg = Float32()
        target_msg.data = float(value_mm)
        self.publisher_.publish(target_msg)
        self.get_logger().info(f'published /target_mm: {target_msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = TargetPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
