import os
from glob import glob
from setuptools import setup

package_name = 'picarx_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Xiaofei Ye',
    maintainer_email='xiaofei.ye.87@gmail.com',
    description='ROS 2 nodes for the PiCar-X self-driving rover '
                '(camera, drive, twist mux, stateful-CNN autopilot, teleop).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_node = picarx_ros.camera_node:main',
            'drive_node = picarx_ros.drive_node:main',
            'odom_node = picarx_ros.odom_node:main',
            'imu_node = picarx_ros.imu_node:main',
            'twist_mux = picarx_ros.twist_mux:main',
            'autopilot_node = picarx_ros.autopilot_node:main',
            'override_teleop = picarx_ros.override_teleop:main',
            'key_teleop = picarx_ros.key_teleop:main',
        ],
    },
)
