# 🤖 Self-Driving Rover

Learning robotics, autonomous vehicles, and physical AI by building a small camera-equipped rover from parts — and teaching it to drive itself.

This repo documents a hands-on project to go from zero to a working self-driving robot, built by a non-engineer with AI as the entire support team. The goal is genuine fluency in how autonomous systems *perceive, decide, and act* — the same mental model behind real AV companies, at a ~$150 scale.

## Demo

The rover driving on its own — camera → trained neural network → steering, running live on a Raspberry Pi:

https://github.com/user-attachments/assets/88f626cc-050c-48b7-9850-fefdfd9f2f26

## The core idea: sense → think → act

Every autonomous machine, from this rover to a robotaxi, runs the same loop dozens of times per second:

- **Sense** — a camera and distance sensor turn the world into data
- **Think** — code (later, a trained neural network) turns data into a driving decision
- **Act** — motors and a steering servo turn the decision into motion

Master that loop on a small rover and the concepts scale directly to full-size autonomy.

## Hardware

- **Brain:** Raspberry Pi 3 Model B (single-board computer)
- **Platform:** SunFounder PiCar-X kit — chassis, drive motor, steering servo, camera, ultrasonic + grayscale line sensors
- **Training compute:** Google Colab (cloud T4 GPU) — no local GPU required

## Roadmap

| Phase | Goal | Key concepts | Status |
|-------|------|--------------|--------|
| 0 · Setup | Tools, repo, order parts | control loop, SBC, version control | ✅ |
| 1 · Make it move | Assemble; drive by code | motors, PWM, servos, actuators, headless control | ✅ |
| 2 · Make it see | Camera + rule-based autonomy | computer vision, OpenCV, line following, obstacle avoidance | ✅ |
| 3 · Make it learn | Train a neural net to drive itself | behavioral cloning, training, edge inference, memory, uncertainty | ✅ |
| 4 · Extend & document | One upgrade (3D printing / depth camera / ROS 2) + write-up | (varies) | ⬜ next |

## Status

🟢 **Phase 3 complete — it drives itself, remembers, recovers, and generalizes to a new track.**

The full physical-AI arc is working end to end: **collect → train → deploy → improve → give it memory → wrap it in a safety net → generalize to a new track.**

1. **Behavioral cloning.** Drive the rover manually to collect ~7k camera frames labeled with steering, then train a **PilotNet** CNN (NVIDIA DAVE-2 architecture) in the cloud and run it on-device with **TF-Lite** to steer around a track in real time (~40 fps on a Pi 3B).
2. **Robustness with data, not knobs.** Diagnosed a mid-corner failure as **regression-to-the-mean on thin data**; fixed it with **DAgger** (targeted correction frames) plus **photometric + flip augmentation** and real daylight data for **domain-shift / lighting** robustness.
3. **Memory.** A purely reactive (single-frame) policy can't commit through a corner once the lane cue leaves the tilted camera at the apex. Gave the car short-term memory with a **frame-stacked** model (last 4 frames as input channels → the same CNN learns motion), deployed via a timestamped ring buffer so the stack's real-time spacing matches training. Result: it now **holds the apex** — the mid-corner give-up is gone.
4. **Uncertainty-aware recovery.** Built a supervisory **fallback layer** over the learned policy: it watches the live steering stream for out-of-distribution behavior (thrash, or a fast collapse of a committed turn), **reverses to reacquire the lane**, and hands control back only once the model is steering the correct direction again — with a fail-safe stop if it can't. Validated on-car over multiple laps.
5. **Generalization to a new track.** Put the memory model — trained only on the original track — on a **brand-new track** with different corners, layout, and lighting. It drove most of it straight from the first track's learning (real **distribution-shift generalization**); only one unfamiliar corner needed a small **targeted DAgger** top-up (~2k frames), after which the car ran **3 laps in each direction**. Adding that data slightly regressed one corner on the *original* track — a textbook **negative-transfer** trade-off in a fixed-capacity model, and a reminder to regression-test the whole operating domain after any change.

## What's in this repo

```
code/
  # Phase 1 — make it move
  smoke_test.py          hardware bring-up: center servos, read sensors, no drive
  first_drive.py         first open-loop timed drive
  calibrate_line.py      calibrate the grayscale line sensors

  # Phase 2 — make it see (rule-based)
  follow_line.py         grayscale edge-following (closed loop)
  avoid_obstacles.py     ultrasonic obstacle avoidance
  camera_test.py         live camera stream check

  # Phase 3 — make it learn (data pipeline)
  collect_data.py        keyboard teleop; log camera frames + steering labels
  inspect_dataset.py     dataset integrity + steering-balance histogram
  clean_dataset.py       flag corrupt / dark / blurry / duplicate frames
  review_flagged.py      contact sheets to eyeball flagged frames
  prune_recovery.py      drop off-track (tape-less) frames via HSV blue-mask
  prune_all.py           whole-dataset prune with a tuned HSV gate
  unprune.py             restore pruned frames (needed once we added memory)
  new_frames_report.py   summarize a fresh DAgger collection
  verify_new.py          auto-detect a fresh DAgger batch (largest time-gap) + QA it
  inspect_sequences.py   temporal analysis: rate, runs, stackable sequences

  # Phase 3 — deploy + diagnose
  autopilot.py           reactive PilotNet autopilot (single frame)
  autopilot_stateful.py  frame-stacked (memory) autopilot
  autopilot_recovery.py  memory autopilot + uncertainty-aware recovery fallback
  occlusion_test.py      tape-vs-background occlusion test (what does the model use?)
  test_recovery_logic.py off-Pi unit tests for the recovery decision logic

notebooks/
  train_pilotnet.ipynb   Colab: build sequences, train, augment, export TF-Lite
```

## Concepts, mapped to real autonomy

The toy scale is deliberate — every problem here has a full-size analog:

- **Calibration** (steering offset, sensor baselines) ↔ AV sensor / extrinsic calibration
- **Open- vs closed-loop control** ↔ why perception closes the loop
- **Behavioral cloning / end-to-end learning** ↔ imitation-learned driving policies
- **Distribution shift & DAgger** ↔ the classic "the policy visits states its data never covered" problem
- **Domain / covariate shift** (night vs daylight) ↔ appearance robustness, augmentation
- **Reactive vs stateful policies** ↔ why memory (temporal context) matters for commitment
- **Epistemic uncertainty & OOD detection** ↔ knowing when the model doesn't know
- **Minimal-risk maneuver / fallback policy** ↔ supervisory safety layers over learned controllers
- **Edge inference** (TF-Lite, quantized, ring buffer) ↔ on-vehicle compute constraints

## How it's built

Written and debugged interactively, using AI as PM, instructor, and engineering support — with the human making the calls, running the hardware, and reading every failure log. Project notes and per-session write-ups live alongside the code.

---

*A learning project — teaching a $150 rover to drive, to learn how real autonomy thinks.*
