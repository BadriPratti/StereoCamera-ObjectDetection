import rclpy
from stereo_processor import StereoProcessor

def main(args=None):
    rclpy.init(args=args)
    node = StereoProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
