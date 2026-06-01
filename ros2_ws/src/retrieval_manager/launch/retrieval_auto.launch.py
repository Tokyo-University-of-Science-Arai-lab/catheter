from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    master_path = LaunchConfiguration("master_path")

    return LaunchDescription([
        DeclareLaunchArgument(
            "master_path",
            default_value="/home/book/pro_book/pro_hand_book_python/master_20260216.json",
            description="Path to retrieval master json"
        ),

        # 外部ノード：出庫リストを順番に送る
        Node(
            package="retrieval_manager",
            executable="retrieval_list_trigger_node",
            name="retrieval_list_trigger_node",
            output="screen",
            parameters=[
                {
                    "master_path": master_path,
                    "initial_wait_sec": 2.0,
                    "after_shelf_id_wait_sec": 0.5,
                    "after_navigation_goal_wait_sec": 0.5,
                }
            ],
        ),

        # リニアリフト高さ制御ノード
        # ros2 run iai_cylinder height_controller と同じ
        Node(
            package="iai_cylinder",
            executable="height_controller",
            name="height_controller",
            output="screen",
        ),
    ])