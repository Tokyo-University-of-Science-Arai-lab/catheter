from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'udp_bridge_manip'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/udp_bridge_manip']),
        ('share/udp_bridge_manip', ['package.xml']),
        (os.path.join('share', 'udp_bridge_manip', 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='book',
    maintainer_email='book@todo.todo',
    description='UDP bridge for manipulator side',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'shelf_id_sender = udp_bridge_manip.shelf_id_sender:main',
            'navigation_goal_receiver = udp_bridge_manip.navigation_goal_receiver:main',
            'navigation_goal_final_receiver = udp_bridge_manip.navigation_goal_final_receiver:main',
            'error_x_sender = udp_bridge_manip.error_x_sender:main',
            'wall_distance_receiver = udp_bridge_manip.wall_distance_receiver:main',
            'wall_yaw_deg_receiver = udp_bridge_manip.wall_yaw_deg_receiver:main',
            'cmd_vel_sender = udp_bridge_manip.cmd_vel_sender:main',
        ],
    },
)
