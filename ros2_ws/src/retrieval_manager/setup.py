from setuptools import setup
import os
from glob import glob

package_name = 'retrieval_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='book',
    maintainer_email='book@example.com',
    description='Retrieval manager package',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'retrieval_list_trigger_node = retrieval_manager.retrieval_list_trigger_node:main',
        ],
    },
)