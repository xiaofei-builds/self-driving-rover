# picarx_ros

ROS 2 (Jazzy) package for a PiCar-X turned into a self-driving rover. It wraps a
Phase-3 vision pipeline (a stateful CNN "pilot" plus a hand-tuned thrash/flip
recovery layer) in a clean node graph with a safety-driver override.

## The node graph

```
camera_node ──/camera/image_raw──▶ autopilot_node ──/cmd_vel_auto──▶ twist_mux ──/cmd_vel──▶ drive_node ──▶ PiCar-X hardware
                                                                        ▲
                                      override_teleop ──/teleop/override┘   (keyboard: guarded nudge / full manual / latched STOP)
```

| Node | What it does |
|------|--------------|
| `camera_node` | Publishes the front camera as `sensor_msgs/Image` at ~15 Hz, 320x240. |
| `autopilot_node` | Runs the 12-channel stateful TFLite pilot + recovery FSM, proposes `Twist` on `/cmd_vel_auto`. Needs the `~/ai` venv (see below). |
| `twist_mux` | Arbiter. Obeys the operator's explicit intent on `/teleop/override` (GUARDED / MANUAL / STOP) and forwards the winning `Twist` to `/cmd_vel`. |
| `drive_node` | Sole owner of the `Picarx` hardware. Ackermann bicycle model, 0.5 s watchdog. `enable_motors:=false` by default (safe). |
| `override_teleop` | Keyboard safety driver. Run in its own terminal. |
| `key_teleop` | Simple direct-drive keyboard teleop (no autopilot), handy for checkout. |

## Two Python interpreters, one graph

`ai_edge_litert` (TFLite) lives only in the `~/ai` virtualenv; the `picarx`
hardware library lives only in system Python. The launch file runs
`autopilot_node` under the venv interpreter (`prefix:=/home/pi/ai/bin/python3`)
and every other node under system Python. DDS bridges them over topics.

## Build (on the Pi)

```bash
mkdir -p ~/ros2_ws/src
cp -r ~/picarx_ros ~/ros2_ws/src/          # or unpack the tarball into src/
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select picarx_ros
source install/setup.bash                  # source this in EVERY new terminal
```

## Run

```bash
# one command, motors OFF (safe dry run -- prints decisions, does not move)
ros2 launch picarx_ros bringup.launch.py

# arm the motors
ros2 launch picarx_ros bringup.launch.py enable_motors:=true

# arm + tune
ros2 launch picarx_ros bringup.launch.py enable_motors:=true gain:=1.5 cruise_mps:=0.06
```

In a **second terminal** (needs its own keyboard/TTY), after sourcing:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run picarx_ros override_teleop
```

### Override keys

`a/d` nudge (guarded) or steer (manual) · `w/s` throttle (grabs full manual) ·
`m` full manual · `p` hand back to autopilot (instant) · `space` latched STOP
(the real e-stop) · `c` centre · `q` quit.

## Hardware notes

The PiCar-X drive motors are open-loop (no wheel encoder), so commanded "m/s" is
a throttle proxy, not a measured speed, and there is a minimum-speed floor below
which the car will not move. Closed-loop speed would need encoders -- a planned
upgrade alongside LiDAR odometry.
