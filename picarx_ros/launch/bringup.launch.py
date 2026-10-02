"""picarx_ros bringup -- start the whole self-driving stack with one command.

Examples
--------
  # motors OFF (safe: nodes run, drive_node prints commands but the car does not move)
  ros2 launch picarx_ros bringup.launch.py

  # arm the motors and drive
  ros2 launch picarx_ros bringup.launch.py enable_motors:=true

  # arm + tune steering gain and cruise speed live
  ros2 launch picarx_ros bringup.launch.py enable_motors:=true gain:=1.5 cruise_mps:=0.06

  # flip the physical steering direction if left/right is reversed (see STEERING SIGN below)
  ros2 launch picarx_ros bringup.launch.py enable_motors:=true drive_steer_sign:=1.0

This starts FOUR background nodes wired camera -> autopilot -> twist_mux -> drive,
PLUS robot_state_publisher, which reads picarx.urdf and publishes the fixed
base_link -> base_laser transform onto /tf_static (see ROBOT DESCRIPTION below).
The keyboard override is deliberately NOT started here: it reads raw keystrokes and
needs its own interactive terminal (a launched node does not own a TTY). Run it in a
second terminal:

  ros2 run picarx_ros override_teleop

ROBOT DESCRIPTION (why robot_state_publisher is here)
-----------------------------------------------------
picarx.urdf declares WHERE the LiDAR sits on the robot: base_laser is 0.063 m forward
of the rear axle, 0.1285 m up, and yawed +90 deg (measured in Session 24 -- the LD19's
zero points to the car's LEFT). robot_state_publisher reads that file and broadcasts the
transform on /tf_static. Everything downstream that needs to place a /scan point in the
robot frame -- RViz, slam_toolbox, Nav2 -- reads that transform. Without it, /scan is a
cloud of ranges with no known origin. NOTE: this reads the urdf from the INSTALLED share
dir, so `colcon build` must have run after any urdf change. All joints are fixed, so no
joint_state_publisher is needed.

STEERING SIGN (why left/right can come up reversed)
---------------------------------------------------
Nothing in this stack MEASURES which way the wheels point -- the direction is an
assumption. Every command (autopilot AND manual) ends at drive_node's yaw->servo
conversion, which multiplies by 'drive_steer_sign'. Flipping that ONE value reverses
both the autopilot and your manual steering together. If, on a fresh deploy, left/right
is mirrored, relaunch with drive_steer_sign:=1.0 (the opposite of the -1.0 default),
watch the wheels, and once correct we bake that value in. A true self-check would need
a sensor (IMU / wheel encoder) -- that is the Phase-4B upgrade.

Why the autopilot uses a different Python (the 'prefix' below)
-------------------------------------------------------------
The TFLite interpreter (ai_edge_litert) is installed ONLY in the ~/ai virtualenv,
not in system Python. The picarx hardware library is only in system Python. So we
run the autopilot node under the venv's interpreter and everything else under system
Python; ROS 2's DDS graph lets the two interpreters talk over topics regardless.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    enable_motors = LaunchConfiguration('enable_motors')
    gain = LaunchConfiguration('gain')
    cruise_mps = LaunchConfiguration('cruise_mps')
    venv_python = LaunchConfiguration('venv_python')
    model_path = LaunchConfiguration('model_path')
    drive_steer_sign = LaunchConfiguration('drive_steer_sign')
    autopilot_steer_sign = LaunchConfiguration('autopilot_steer_sign')
    odom_yaw_sign = LaunchConfiguration('odom_yaw_sign')
    imu_yaw_sign = LaunchConfiguration('imu_yaw_sign')

    declared_args = [
        DeclareLaunchArgument(
            'enable_motors', default_value='false',
            description='Arm the drive motors. false = safe: drive_node computes and '
                        'prints commands but does not spin the motors.'),
        DeclareLaunchArgument(
            'gain', default_value='1.5',
            description='Autopilot steering gain -- scales the CNN steering output.'),
        DeclareLaunchArgument(
            'cruise_mps', default_value='0.06',
            description='Autopilot forward cruise speed proposal in m/s (open-loop, '
                        'no encoder -- this is a commanded throttle, not a measured speed).'),
        DeclareLaunchArgument(
            'venv_python', default_value='/home/pi/ai/bin/python3',
            description='Interpreter that has ai_edge_litert (the ~/ai venv), used to '
                        'run the autopilot node.'),
        DeclareLaunchArgument(
            'model_path', default_value='/home/pi/pilot_stateful.tflite',
            description='Path to the 12-channel stateful pilot .tflite model.'),
        DeclareLaunchArgument(
            'drive_steer_sign', default_value='1.0',
            description='Sign of drive_node yaw->servo conversion. Flip to 1.0 if BOTH '
                        'autopilot and manual steering come up left/right reversed.'),
        DeclareLaunchArgument(
            'autopilot_steer_sign', default_value='1.0',
            description='Sign folding CNN steering polarity into the published yaw. Change '
                        'only if the autopilot alone steers the wrong way vs. manual.'),
        DeclareLaunchArgument(
            'odom_yaw_sign', default_value='-1.0',
            description='Sign applied to /cmd_vel angular.z when integrating odometry. '
                        'CONFIRMED -1.0 on this car (S26): /cmd_vel angular.z is inverted '
                        'vs ROS +z=CCW=LEFT, so odom yaw rises on a left turn only at -1.0.'),
        DeclareLaunchArgument(
            'imu_yaw_sign', default_value='1.0',
            description='Sign applied to BNO085 gyro z when publishing /imu/data. '
                        'S29 measured +1.0 in the mounted pose: a left/CCW turn '
                        'reads gz positive, already matching REP-103 (+z=CCW=LEFT).'),
    ]

    # robot_description: read the installed URDF text once at launch-generation time.
    # get_package_share_directory resolves to the colcon-INSTALLED share dir, so the urdf
    # must have been built in (setup.py data_files). All joints fixed -> no
    # joint_state_publisher.
    urdf_path = os.path.join(
        get_package_share_directory('picarx_ros'), 'urdf', 'picarx.urdf')
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher', output='screen',
        parameters=[{'robot_description': robot_description}],
    )

    # Launch arguments arrive as STRINGS. ROS parameters are typed, so we coerce each
    # one to the type the node declared (bool / float) with ParameterValue.
    camera = Node(
        package='picarx_ros', executable='camera_node', name='camera_node',
        output='screen',
    )

    autopilot = Node(
        package='picarx_ros', executable='autopilot_node', name='autopilot_node',
        output='screen',
        prefix=venv_python,   # <-- run this node under the ~/ai venv interpreter
        parameters=[{
            'gain': ParameterValue(gain, value_type=float),
            'cruise_mps': ParameterValue(cruise_mps, value_type=float),
            'model_path': ParameterValue(model_path, value_type=str),
            'steer_sign': ParameterValue(autopilot_steer_sign, value_type=float),
        }],
    )

    twist_mux = Node(
        package='picarx_ros', executable='twist_mux', name='twist_mux',
        output='screen',
    )

    drive = Node(
        package='picarx_ros', executable='drive_node', name='drive_node',
        output='screen',
        parameters=[{
            'enable_motors': ParameterValue(enable_motors, value_type=bool),
            'steer_sign': ParameterValue(drive_steer_sign, value_type=float),
        }],
    )

    # odom_node: dead-reckoning command odometry (Phase 5, N1). Integrates
    # /cmd_vel into the odom -> base_link transform. Lives in bringup for now;
    # will move to a localization launch when we assemble the nav stack.
    odom = Node(
        package='picarx_ros', executable='odom_node', name='odom_node',
        output='screen',
        parameters=[{
            'yaw_sign': ParameterValue(odom_yaw_sign, value_type=float),
        }],
    )

    # imu_node: BNO085 gyro-only -> /imu/data (Phase 5, N1). Publishes the yaw
    # rate the EKF needs; command-odom cannot give a trustworthy heading (S27).
    # System python (adafruit libs are --user installed there) -> no prefix.
    imu = Node(
        package='picarx_ros', executable='imu_node', name='imu_node',
        output='screen',
        parameters=[{
            'imu_yaw_sign': ParameterValue(imu_yaw_sign, value_type=float),
        }],
    )

    return LaunchDescription(
        declared_args + [robot_state_publisher, camera, autopilot, twist_mux, drive, odom, imu])
