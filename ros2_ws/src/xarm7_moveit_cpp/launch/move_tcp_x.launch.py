from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():

    # ============================
    # xArm MoveIt (real robot)
    # ============================
    xarm_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("xarm_moveit_config"),
                "launch",
                "_robot_moveit_realmove.launch.py"
            ])
        ),
        launch_arguments={
            "robot_ip": "192.168.1.208",
            "dof": "7",
            "robot_type": "xarm",
            "hw_ns": "xarm",
            "no_gui_ctrl": "false",
            "use_rviz": "false",
        }.items(),
    )

    # ============================
    # Your C++ node
    # ============================
    move_tcp = Node(
        package="xarm7_moveit_cpp",
        executable="move_tcp_x",
        name="move_tcp_minus_x",
        output="screen",
    )

    return LaunchDescription([
        xarm_moveit,
        #move_tcp
    ])
