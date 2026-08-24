#!/usr/bin/env python3
"""
autopilot_node.py — M6 of the Phase 4A ROS 2 migration: the PERCEPTION+POLICY node.

This is the ROS version of the Phase-3 self-driving pipeline (autopilot_recovery.py),
split correctly for a robot graph:

    camera_node  --/camera/image_raw-->  [autopilot_node]  --/cmd_vel_auto-->  twist_mux --> drive_node

autopilot_node is a PURE POLICY. It looks at camera frames and *proposes* a driving
command. It never touches hardware — no Picarx(), no servo, no motors. drive_node is
still the sole owner of the I2C bus (the single-writer rule we set in M2). All this node
does is publish a geometry_msgs/Twist on /cmd_vel_auto and let the mux + drive_node decide
whether that proposal actually reaches the wheels.

WHAT IT PORTS FROM PHASE 3 (unchanged in spirit):
  * The STATEFUL CNN. The model takes a stack of N=4 frames spaced DT_STRIDE=4/15 s apart
    in REAL time (picked from a timestamped ring buffer, not by loop count), each resized
    to 160x120, BGR->RGB, /255 float32. Input tensor (1,120,160,12). Output x30 = degrees.
    This preprocessing is BYTE-IDENTICAL to training — a model is only as valid as the
    pixels you feed it match the pixels it learned on.
  * The RECOVERY self-monitor (S11/S12). The node watches its OWN steering output for
    out-of-distribution "thrash" (high variance / sign-flips) or a fast collapse of a
    committed turn ("flip"). When it trips, the node stops proposing forward motion and
    instead proposes a slow straight REVERSE, watching its output re-stabilise before
    handing itself back to DRIVE. This is a policy monitoring its own uncertainty — the
    seed of the "camera proposes, an independent check vetoes" story that the LiDAR layer
    will complete.

WHAT CHANGED vs autopilot_recovery.py, and WHY:
  * No grayscale hint. In Phase 3 the recovery could peek at the down-facing grayscale
    sensors to hand back sooner. Those sensors live on the I2C bus that drive_node now
    owns exclusively, so this node cannot read them without breaking the single-writer
    rule. The grayscale was only ever an ACCELERATOR (it never gated handback — CNN
    re-confidence was always the real criterion), so dropping it costs a little handback
    latency, nothing correctness. If we want it back later, the clean move is for
    drive_node to publish /picarx/grayscale and this node to subscribe — a new wire, not
    a second bus owner.
  * Reverse/stop are now COMMANDS (negative linear.x, zero angular.z) published on
    /cmd_vel_auto, not direct px.backward() calls. Same behaviour, expressed through the
    graph.

Run (on the Pi). Nothing moves from THIS node — motors are drive_node's call:
    python3 autopilot_node.py
    python3 autopilot_node.py --ros-args -p enable_recovery:=false      # plain policy
    python3 autopilot_node.py --ros-args -p cruise_mps:=0.06 -p gain:=1.2

Topics
------
subscribes : /camera/image_raw   sensor_msgs/Image   (bgr8, BEST_EFFORT — see QoS note)
publishes  : /cmd_vel_auto       geometry_msgs/Twist (the PROPOSAL)
             /autopilot/state    std_msgs/String      (WARMUP|DRIVE|RECOVER|STOPPED — for Foxglove)

Session 17 of the self-driving-rover project.
"""

import math
import time
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String


# ============================================================================
# PURE OOD-DETECTION HELPERS
# Copied verbatim from autopilot_recovery.py (Session 12), where they are unit-
# tested in test_recovery_logic.py. numpy only — no hardware, no ROS. Kept in
# this file so the node is self-contained until M7 packages everything.
# ============================================================================

def thrash_stats(window, deadband=2.0):
    """(std, sign_flips) over a window of recent steering angles (deg).
    sign_flips ignores |angle| < deadband so straight-ahead jitter doesn't count."""
    a = np.asarray(window, dtype=np.float32)
    if a.size < 2:
        return 0.0, 0
    std = float(np.std(a))
    signs = np.sign(np.where(np.abs(a) < deadband, 0.0, a))
    nz = signs[signs != 0.0]
    flips = int(np.sum(nz[1:] != nz[:-1])) if nz.size >= 2 else 0
    return std, flips


def is_confused(hist, W, std_hi, flip_hi, deadband=2.0):
    """True if the last W outputs look like OOD thrash: high spread OR many flips."""
    if len(hist) < W:
        return False
    std, flips = thrash_stats(list(hist)[-W:], deadband)
    return (std > std_hi) or (flips >= flip_hi)


def is_calm(hist, W, std_lo, flip_lo, deadband=2.0):
    """True if the last W outputs look settled: low spread AND few flips.
    std_lo<std_hi and flip_lo<flip_hi give hysteresis (no chatter on the boundary)."""
    if len(hist) < W:
        return False
    std, flips = thrash_stats(list(hist)[-W:], deadband)
    return (std < std_lo) and (flips <= flip_lo)


def flip_check(window, mag=8.0, swing=12.0, min_side=3):
    """Detect a FAST, LARGE collapse of a committed turn (a confident reversal OR a
    mid-corner give-up). Returns (fired, committed_dir); committed_dir is the sign of
    the abandoned turn = the direction the CNN must return to before handback.
    See autopilot_recovery.py for the full rationale."""
    a = np.asarray(window, dtype=np.float32)
    n = a.size
    if n < 2 * min_side:
        return False, 0.0
    for k in range(min_side, n - min_side + 1):
        old = float(a[:k].mean())
        new = float(a[k:].mean())
        if abs(old) > mag:
            s = 1.0 if old > 0 else -1.0
            if (old - new) * s > swing:
                return True, s
    return False, 0.0


# ============================================================================
# THE NODE
# ============================================================================

class AutopilotNode(Node):

    def __init__(self):
        super().__init__('autopilot_node')

        # ---- model + timing (must match TRAINING; these are not free knobs) ----
        self.declare_parameter('model_path', '/home/pi/pilot_stateful.tflite')
        self.declare_parameter('n_frames', 4)          # stack depth the model expects
        self.declare_parameter('dt_stride', 4.0 / 15.0)  # real-time spacing between stacked frames
        self.declare_parameter('img_w', 160)           # model input width  (resize target)
        self.declare_parameter('img_h', 120)           # model input height
        self.declare_parameter('infer_rate_hz', 15.0)  # how often we run the net

        # ---- driving command shaping ----
        self.declare_parameter('gain', 1.2)            # scale CNN steering (S11 used 1.2)
        self.declare_parameter('cruise_mps', 0.05)     # forward speed the node proposes
        self.declare_parameter('rev_mps', 0.03)        # reverse speed during RECOVER
        self.declare_parameter('max_steer_deg', 30.0)  # clamp, matches the car
        self.declare_parameter('wheelbase_m', 0.094)   # MUST match drive_node
        # steer_sign folds the SunFounder servo polarity into the published yaw rate, the
        # SAME way key_teleop/joy_teleop do, so every command producer speaks one language
        # to drive_node. The CNN's raw output is already in servo convention (+ve = RIGHT,
        # because in Phase 3 it fed set_dir_servo_angle directly). Plugging that value in
        # here with steer_sign=-1 makes drive_node reproduce exactly that servo angle. See
        # steer_to_twist() for the one line of algebra. Sign is only ever verified by
        # WATCHING THE WHEELS — if autopilot steers mirror-image, flip this one number.
        self.declare_parameter('steer_sign', 1.0)
        self.declare_parameter('image_topic', 'camera/image_raw')

        # ---- recovery self-monitor (grayscale-free; CNN re-confidence is the criterion) ----
        self.declare_parameter('enable_recovery', True)
        self.declare_parameter('win', 8)               # thrash window (# recent outputs)
        self.declare_parameter('std_hi', 12.0)         # std(deg) above = confused
        self.declare_parameter('flip_hi', 3)           # sign-flips in window above = confused
        self.declare_parameter('std_lo', 5.0)          # std below = calm (hysteresis)
        self.declare_parameter('flip_lo', 1)           # flips at/below = calm
        self.declare_parameter('confuse_ticks', 8)     # consecutive THRASH ticks to trip
        self.declare_parameter('flip_mag', 8.0)        # |steer| the abandoned turn must have reached
        self.declare_parameter('flip_swing', 12.0)     # deg the committed turn must collapse to trip
        self.declare_parameter('flip_ticks', 2)        # consecutive flip ticks to trip (fires fast)
        self.declare_parameter('reacquire_ticks', 5)   # consecutive calm ticks to resume DRIVE
        self.declare_parameter('rev_max_s', 2.5)       # MAX reverse before fail-safe STOP (no rear sensor)

        g = self.get_parameter
        self.N = int(g('n_frames').value)
        self.DT = float(g('dt_stride').value)
        self.SPAN = (self.N - 1) * self.DT
        self.W = int(g('img_w').value)
        self.H = int(g('img_h').value)
        self.infer_rate = float(g('infer_rate_hz').value)
        self.gain = float(g('gain').value)
        self.cruise = float(g('cruise_mps').value)
        self.rev = float(g('rev_mps').value)
        self.max_steer = float(g('max_steer_deg').value)
        self.L = float(g('wheelbase_m').value)
        self.steer_sign = float(g('steer_sign').value)
        self.enable_recovery = bool(g('enable_recovery').value)
        self.win = int(g('win').value)
        self.std_hi = float(g('std_hi').value)
        self.flip_hi = int(g('flip_hi').value)
        self.std_lo = float(g('std_lo').value)
        self.flip_lo = int(g('flip_lo').value)
        self.confuse_ticks = int(g('confuse_ticks').value)
        self.flip_mag = float(g('flip_mag').value)
        self.flip_swing = float(g('flip_swing').value)
        self.flip_ticks = int(g('flip_ticks').value)
        self.reacquire_ticks = int(g('reacquire_ticks').value)
        self.rev_max = float(g('rev_max_s').value)
        image_topic = str(g('image_topic').value)

        # ---- load the model + cv2 (deferred so this file imports clean off-Pi) ----
        # Same discipline as camera_node.py: heavy / hardware-ish imports live here, not
        # at module top, so the node can be opened and syntax-checked on the Surface.
        import cv2
        from ai_edge_litert.interpreter import Interpreter
        self.cv2 = cv2

        model_path = str(g('model_path').value)
        self.itp = Interpreter(model_path=model_path)
        self.itp.allocate_tensors()
        self.inp = self.itp.get_input_details()[0]
        self.out = self.itp.get_output_details()[0]
        expected = (1, self.H, self.W, self.N * 3)
        got = tuple(int(x) for x in self.inp['shape'])
        if got != expected:
            raise RuntimeError(
                f'model input {got} != expected {expected}. Wrong .tflite? '
                f'This node expects the 12-channel stateful pilot.')

        # ---- state ----
        self.buf = deque()             # (t_sec, frame_rgb_float32 HxWx3) oldest..newest
        self.hist = deque(maxlen=self.win)   # recent commanded steering (deg)
        self.state = 'WARMUP'
        self.confuse_ct = 0
        self.flip_ct = 0
        self.calm_ct = 0
        self.recover_dir = 0.0         # after a FLIP: sign the CNN must return to
        self.rev_start = None
        self.last_log = 0.0

        # ---- ROS interface ----
        # QoS: the camera publishes SENSOR_DATA (BEST_EFFORT). A default (RELIABLE)
        # subscription would silently receive NOTHING — the M4 QoS trap. Match it.
        self.create_subscription(
            Image, image_topic, self.on_image, qos_profile_sensor_data)
        # /cmd_vel_auto is a control command: default QoS (RELIABLE, depth 10), same as
        # teleop's /cmd_vel_manual, so the mux sees both inputs on equal contracts.
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel_auto', 10)
        self.state_pub = self.create_publisher(String, 'autopilot/state', 10)

        # Inference on a FIXED timer, not per-frame: the control rate is decoupled from
        # the camera rate, and the stateful stack is always sampled by real timestamps
        # from the buffer regardless of jitter. Same fixed-rate philosophy as drive_node.
        self.create_timer(1.0 / self.infer_rate, self.tick)

        self.get_logger().info(
            f'autopilot_node up | model OK {got} | infer={self.infer_rate:.0f}Hz '
            f'cruise={self.cruise:.3f}m/s gain={self.gain:.2f} '
            f'steer_sign={self.steer_sign:+.0f} recovery={self.enable_recovery}')
        self.get_logger().warn(
            'This node only PROPOSES on /cmd_vel_auto. Nothing moves until the mux '
            'forwards it and drive_node runs with enable_motors:=true.')

    # ------------------------------------------------------------------------
    def on_image(self, msg: Image):
        """Turn an incoming frame into a preprocessed RGB float and buffer it with its
        real capture time. Preprocessing MUST match training exactly."""
        # Reconstruct the pixel array. camera_node publishes bgr8; accept rgb8 too.
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
        except ValueError:
            self.get_logger().warn('image reshape failed — bad step/encoding?')
            return
        if msg.encoding == 'rgb8':
            bgr = arr[:, :, ::-1]
        else:                                    # 'bgr8' (or anything OpenCV-native)
            bgr = arr

        # Resize to the model's input, THEN BGR->RGB, /255. Identical to Phase-3
        # preprocess(): r = cv2.resize(bgr,(W,H)); (r[:,:,::-1]/255).
        small = self.cv2.resize(bgr, (self.W, self.H))
        rgb = (small[:, :, ::-1].astype(np.float32)) / 255.0

        # Timestamp from the message header (the camera's capture time), not "now" —
        # the whole point of the ring buffer is real-time spacing, and the header stamp
        # is the honest time the pixels were taken.
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t == 0.0:                             # header not stamped for some reason
            t = time.time()
        self.buf.append((t, rgb))
        while self.buf and t - self.buf[0][0] > self.SPAN + 0.3:
            self.buf.popleft()

    # ------------------------------------------------------------------------
    def pick_stack(self, now):
        """Pick N frames from the buffer at now, now-DT, now-2*DT, ... by REAL time
        (nearest match), oldest-first, and concat along channels -> (1,H,W,3N)."""
        picks = []
        for k in range(self.N - 1, -1, -1):
            target = now - k * self.DT
            _, fr = min(self.buf, key=lambda tfr: abs(tfr[0] - target))
            picks.append(fr)
        return np.concatenate(picks, axis=2)[None].astype(np.float32)

    def cnn_steer(self, now):
        """Run the stateful net -> steering angle in SERVO convention (deg), gained+clamped."""
        x = self.pick_stack(now)
        self.itp.set_tensor(self.inp['index'], x)
        self.itp.invoke()
        raw = float(self.itp.get_tensor(self.out['index'])[0][0]) * 30.0
        return max(-self.max_steer, min(self.max_steer, raw * self.gain))

    def steer_to_twist(self, v, steer_deg):
        """Command shaping shared with teleop. Inverse bicycle model turns a steering
        ANGLE into a yaw rate at speed v, and steer_sign folds in the servo polarity:

            w = steer_sign * v * tan(steer) / L

        drive_node then inverts this (delta = deg(atan(L*w/v)) * steer_sign) and, with
        steer_sign=-1 on both ends, lands the servo back on exactly `steer_deg`. So the
        CNN's servo-convention output arrives at the servo unchanged. At v=0, w=0 — a
        stationary car has no yaw rate, which is physically correct."""
        w = self.steer_sign * v * math.tan(math.radians(steer_deg)) / self.L
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        return msg

    def publish_state(self):
        s = String()
        s.data = self.state
        self.state_pub.publish(s)

    # ------------------------------------------------------------------------
    def tick(self):
        """Fixed-rate: run the policy + recovery FSM, publish a proposal on /cmd_vel_auto."""
        now = time.time()

        # Not enough real history yet -> propose a safe stop and stay in WARMUP.
        # Publishing zeros (not silence) keeps /cmd_vel_auto ALIVE so the mux can tell
        # "autopilot present, asking to stop" apart from "autopilot dead".
        if len(self.buf) < self.N or (self.buf[-1][0] - self.buf[0][0]) < self.SPAN * 0.9:
            self.state = 'WARMUP'
            self.cmd_pub.publish(self.steer_to_twist(0.0, 0.0))
            self.publish_state()
            return

        # buffer times are the camera stamps; sample the stack at the newest stamp.
        t_ref = self.buf[-1][0]
        steer = self.cnn_steer(t_ref)
        self.hist.append(steer)

        if not self.enable_recovery:
            # Plain policy: always DRIVE, forward at cruise, steer as the net says.
            self.state = 'DRIVE'
            self.cmd_pub.publish(self.steer_to_twist(self.cruise, steer))
            self.publish_state()
            self._maybe_log(now, steer)
            return

        # ---------------- recovery state machine ----------------
        if self.state in ('WARMUP',):
            self.state = 'DRIVE'

        if self.state == 'DRIVE':
            # Propose forward + CNN steering.
            self.cmd_pub.publish(self.steer_to_twist(self.cruise, steer))

            # Two independent confusion triggers, each debounced on its own timescale.
            self.confuse_ct = self.confuse_ct + 1 if is_confused(
                self.hist, self.win, self.std_hi, self.flip_hi) else 0
            flipped, flip_dir = flip_check(
                list(self.hist), self.flip_mag, self.flip_swing)
            self.flip_ct = self.flip_ct + 1 if flipped else 0

            trip = None
            if self.confuse_ct >= self.confuse_ticks:
                trip = 'THRASH'
            elif self.flip_ct >= self.flip_ticks:
                trip = 'FLIP'

            if trip:
                std, flips = thrash_stats(list(self.hist), 2.0)
                if trip == 'FLIP':
                    self.recover_dir = flip_dir
                    self.get_logger().warn(
                        'CONFUSED[FLIP] -> RECOVER (reverse straight); correct dir = %s'
                        % ('LEFT' if self.recover_dir < 0 else 'RIGHT'))
                else:
                    self.recover_dir = 0.0
                    self.get_logger().warn(
                        'CONFUSED[THRASH] (std=%.1f flips=%d) -> RECOVER (reverse straight)'
                        % (std, flips))
                self.state = 'RECOVER'
                self.rev_start = now
                self.calm_ct = 0
                self.confuse_ct = 0
                self.flip_ct = 0
                self.hist.clear()      # measure re-stabilisation on FRESH reverse-phase outputs

        elif self.state == 'RECOVER':
            # Propose a slow STRAIGHT reverse (simpler + safer with no rear sensor).
            self.cmd_pub.publish(self.steer_to_twist(-self.rev, 0.0))

            cnn_ready = is_calm(self.hist, self.win, self.std_lo, self.flip_lo)
            # After a FLIP, don't hand back to a still-confidently-WRONG model: require
            # its steering to point back to the abandoned (correct) direction. THRASH has
            # no single correct side (recover_dir == 0 -> always dir_ok).
            dir_ok = (self.recover_dir == 0.0) or (steer * self.recover_dir > 0.0)
            self.calm_ct = self.calm_ct + 1 if (cnn_ready and dir_ok) else 0

            if self.calm_ct >= self.reacquire_ticks:
                self.get_logger().info('REACQUIRED -> DRIVE')
                self.state = 'DRIVE'
                self.confuse_ct = 0
                self.flip_ct = 0
                self.hist.clear()
            elif now - self.rev_start > self.rev_max:
                self.get_logger().warn(
                    'FAIL-SAFE: reversed %.1fs without recovering -> STOPPED' % self.rev_max)
                self.state = 'STOPPED'

        elif self.state == 'STOPPED':
            # Terminal fail-safe: keep proposing a full stop. A human must take over via
            # the mux (grab teleop) to get moving again — we will NOT silently re-arm.
            self.cmd_pub.publish(self.steer_to_twist(0.0, 0.0))

        self.publish_state()
        self._maybe_log(now, steer)

    # ------------------------------------------------------------------------
    def _maybe_log(self, now, steer):
        if now - self.last_log >= 0.5:
            std, flips = thrash_stats(list(self.hist), 2.0)
            self.get_logger().info(
                '[%s] steer=%+6.1f std=%4.1f flips=%d' % (self.state, steer, std, flips))
            self.last_log = now

    def shutdown(self):
        # Leave the graph in a safe proposal: stop. drive_node's own watchdog also
        # covers us if this node dies outright.
        try:
            self.cmd_pub.publish(self.steer_to_twist(0.0, 0.0))
        except Exception:
            pass


def main():
    rclpy.init()
    node = AutopilotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
