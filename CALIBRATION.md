# Calibration values

Servo trim offsets found during Phase 1 calibration
(`~/picar-x/example/1.cali_servo_motor.py`), stored on the Pi at
`~/.config/picar-x/picar-x.conf`. Recorded here so they survive a reflash.

| Servo | Channel | Offset |
|---|---|---|
| Steering | P2 | **-8.4** |
| Camera pan | P0 | **-12.0** |
| Camera tilt | P1 | **2.0** |

## What an offset is
A servo's software zero rarely equals true mechanical straight (the arm sits a
tooth off, the linkage has slack). Calibration measures that gap once and stores an
offset the code adds to every command, so `set_dir_servo_angle(0)` really points
straight. Same principle as extrinsic calibration on a real AV's camera/LiDAR:
measured pose vs. true mounted pose.

## How steering was verified
Open-loop straight-line drive (`code/first_drive.py`), measuring lateral drift over
~0.6 m. Offset -5.2 drifted ~3 cm right; -8.4 tracked straight. Open-loop drift can
be trimmed but never fully zeroed - that is the job of closed-loop control
(Phase 2 vision), not calibration.

## Grayscale line sensors (Session 4)
- FLOOR baseline: [981.6, 1109.4, 1222.7]
- TAPE  baseline: [1196.6, 1349.9, 1410.4]
- Note: tape reads HIGHER than floor on this surface (blue semi-gloss tape on light wood).
  Re-run calibrate_line.py if the surface or lighting changes.
