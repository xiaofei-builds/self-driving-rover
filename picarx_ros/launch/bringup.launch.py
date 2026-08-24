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

This starts FOUR background nodes wired camera -> autopilot -> twist_mux -> drive.
The keyboard override is deliberately NOT started here: it reads raw keystrokes and
needs its own interactive terminal (a launched node does not own a TTY). Run it in a
second terminal:

  ros2 run picarx_ros override_teleop

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
    ]

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

    return LaunchDescription(declared_args + [camera, autopilot, twist_mux, drive])
