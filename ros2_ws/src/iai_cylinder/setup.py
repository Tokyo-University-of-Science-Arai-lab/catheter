from setuptools import find_packages, setup

package_name = 'iai_cylinder'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shoseki',
    maintainer_email='simazakishota@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
               'height_controller = iai_cylinder.height_controller:main',
               'iai_cylinder_alarm = iai_cylinder.iai_cylinder_alarm:main',
               'iai_cylinder_home = iai_cylinder.iai_cylinder_home:main',

        ],
    },
)
