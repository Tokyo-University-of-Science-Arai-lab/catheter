from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        Node(
            package='udp_bridge_manip',
            executable='shelf_id_sender',
            name='shelf_id_sender',
            output='screen'
        ),

        Node(
            package='udp_bridge_manip',
            executable='navigation_goal_receiver',
            name='navigation_goal_receiver',
            output='screen'
        ),

        Node(
            package='udp_bridge_manip',
            executable='navigation_goal_final_receiver',
            name='navigation_goal_final_receiver',
            output='screen'
        ),

        Node(
            package='udp_bridge_manip',
            executable='error_x_sender',
            name='error_x_sender',
            output='screen'
        ),

        Node(
            package='udp_bridge_manip',
            executable='wall_distance_receiver',
            name='wall_distance_receiver',
            output='screen'
        ),

        Node(
            package='udp_bridge_manip',
            executable='wall_yaw_deg_receiver',
            name='wall_yaw_deg_receiver',
            output='screen'
        ),

        Node(
            package='udp_bridge_manip',
            executable='cmd_vel_sender',
            name='cmd_vel_sender',
            output='screen'
        )
    ])
