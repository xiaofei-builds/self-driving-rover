# Phase 4 · Production-ize with ROS 2

Phase 3 got the rover *driving itself*. It also left a pile of standalone Python scripts you started by hand, in the right order, across five terminals, with two different Python interpreters and a set of flags you had to remember. That is fine for a prototype and fatal for anything you'd hand to someone else. Phase 4 is the "make it real" phase: rebuild the same **sense → think → act** loop on **ROS 2**, and package it so the whole rover comes up with one command.

## Why ROS 2 at all

ROS 2 (Robot Operating System 2) is the de-facto middleware for real robots. The one idea to take away: a robot's software is split into small independent programs called **nodes**, each doing one job, that talk to each other by publishing and subscribing to named **topics**. A node doesn't know or care who else is running — it just puts messages on a topic and reads messages off others. Underneath, ROS 2 uses **DDS** (a pub/sub transport) to move those messages, with **QoS** (quality-of-service) settings that decide, e.g., whether a stream is reliable or best-effort.

Why this matters beyond the toy: decomposition into nodes + topics is *exactly* how full-size autonomy stacks are built. Perception, localization, planning, and control are separate processes exchanging messages, so any one can be swapped, tested, recorded, or replaced without touching the others.

## The node graph

```
camera_node ──/camera/image_raw──▶ autopilot_node ──/cmd_vel_auto──▶ twist_mux ──/cmd_vel──▶ drive_node ──▶ PiCar-X
                                                                        ▲
                                    override_teleop ──/teleop/override──┘   (safety driver)
```

| Node | Job |
|------|-----|
| `camera_node` | Turns the camera into a `sensor_msgs/Image` stream at ~15 Hz. **Sense.** |
| `autopilot_node` | Runs the stateful CNN pilot + the recovery layer, publishes a proposed `Twist`. **Think.** |
| `twist_mux` | The arbiter — decides whether the autopilot or the human operator is in command. |
| `drive_node` | Sole owner of the hardware; converts the winning `Twist` to servo + motor. **Act.** |
| `override_teleop` | The human safety driver's keyboard console (its own terminal). |

## Milestones

- **M1–M3 — foundation.** Installed ROS 2 Jazzy on the Pi. Wrote `drive_node`, the single node allowed to touch the motor/servo hardware (everything else must go through it — one owner of the I2C bus avoids conflicts). It takes a `geometry_msgs/Twist` velocity command and converts it to a front-wheel steering angle with the **Ackermann bicycle model**, with a 0.5 s watchdog that stops the car if commands stop arriving. Added keyboard teleop to drive it by hand.
- **M4 — the camera node.** Publishes frames at 15 Hz. First real encounter with **QoS**: the camera publishes best-effort ("sensor data" profile); a subscriber using the default reliable profile silently receives *nothing*. Matching QoS is a genuine ROS gotcha and a good lesson in why the transport contract matters.
- **M5 — recording.** Used `rosbag2` to record live drives to **MCAP** files, viewable offline in Foxglove. Recording is a separate concern — a built-in tool subscribing to topics, not a node we wrote — which is the point: because everything is a topic, you can capture the exact inputs and outputs of a drive for later analysis without changing the driving code.
- **M6 — autopilot + safety arbiter.** Wrapped the Phase-3 CNN and its uncertainty-aware recovery layer as `autopilot_node`, which *proposes* steering on `/cmd_vel_auto`. Then built `twist_mux`, an **arbiter** that gives a real safety-driver override with three explicit modes:
  - **GUARDED** — the autopilot drives, but the operator can nudge the steering on top of it (shared control on separate axes).
  - **MANUAL** — the operator takes full control of steering and throttle.
  - **STOP** — a latched emergency stop (the real e-stop).

  The design lesson: the first version tried to *infer* whether the human was driving from the command values, which is ambiguous (a zero command could mean "idle" or "hold straight"). The fix was to make the operator send **explicit intent** instead of guessing — the same principle behind a real disengage button.
- **M7 — packaging.** Turned the six scripts into a proper **colcon** package, `picarx_ros`, with a `package.xml` manifest, declared dependencies, console-script entry points for every node, and a single **launch file**. The whole rover now starts with:

  ```bash
  ros2 launch picarx_ros bringup.launch.py enable_motors:=true
  ```

  (Motors are **off by default** — a plain launch is a safe dry run that prints decisions without moving.)

## Two problems worth understanding

**1. One graph, two Python interpreters.** The CNN runs through a TFLite interpreter that only exists inside a Python virtualenv (`~/ai`). The hardware library only exists in system Python. They can't both be the "one" Python. The launch file solves it by running just the autopilot node under the venv's interpreter (via the launch `prefix`), while every other node runs under system Python — and ROS 2's DDS graph lets the two interpreters exchange messages as if nothing were unusual. This is a small version of a real production headache: a perception node that needs CUDA/TensorRT libraries a control node doesn't, all in one robot.

**2. Sign conventions with no ground truth.** On the first packaged deploy, left and right were mirrored — for both the autopilot *and* manual steering. The root cause is worth internalizing: **nothing in the stack measures which way the wheels actually point.** The direction is an assumption, and the final left/right is the product of four ±1 sign choices in series (the CNN's convention, the bicycle-model yaw definition, the operator convention, and the drive node's servo polarity). Change anything — including how the steering servo is physically mounted or calibrated — and the whole thing flips, and no software can catch it. Because both the autopilot and manual steering reversed *together*, the culprit was the one stage they share: the drive node's yaw-to-servo conversion. We calibrated it by watching the wheels, then **persisted** the confirmed value (and exposed the signs as launch arguments so it's a ten-second fix, not a mystery, if the hardware changes again).

The deeper takeaway: an unverified convention is a recurring coin-flip. The durable fixes are (a) calibrate once and persist, and (b) add a sensor — an IMU or wheel encoder — so the robot can *check its own* steering and speed instead of trusting an open-loop assumption. That "no feedback" gap is the motivation for the next phase.

## The package

```
picarx_ros/
  package.xml          manifest: name, version, deps (rclpy, sensor_msgs, geometry_msgs, std_msgs)
  setup.py             console-script entry points (one runnable command per node)
  setup.cfg
  resource/picarx_ros
  launch/
    bringup.launch.py  declares the node graph + typed launch arguments
  picarx_ros/
    camera_node.py  drive_node.py  twist_mux.py
    autopilot_node.py  override_teleop.py  key_teleop.py
```

Build and run on the Pi:

```bash
# one-time: create a workspace, drop the package in its src/, build
mkdir -p ~/ros2_ws/src && cp -r picarx_ros ~/ros2_ws/src/
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select picarx_ros
source install/setup.bash

# bring up the rover (motors off = safe dry run; add enable_motors:=true to drive)
ros2 launch picarx_ros bringup.launch.py

# the human safety driver runs in its OWN terminal (it needs an interactive keyboard):
ros2 run picarx_ros override_teleop
```

The safety driver is deliberately *not* part of the launch file — it reads raw keystrokes and needs to own an interactive terminal, and keeping the human override as a separate process mirrors how real safety-driver controls are kept independent of the autonomy computer.

## What's next

- Put this repo history in order (done — you're reading it).
- **LiDAR + odometry (Phase 4B / 5).** Add a distance sensor and, crucially, a sensor that gives *feedback* on motion (IMU / encoder), so the rover can localize, map (SLAM), and check its own steering/speed instead of trusting open-loop assumptions. The eventual demo: *"camera proposes, LiDAR vetoes"* — an independent safety monitor that can override the learned policy, using the same arbiter socket built in M6.
