# 🤖 Self-Driving Rover

Learning robotics, autonomous vehicles, and physical AI by building a small camera-equipped rover from parts — and teaching it to drive itself.

This repo documents a hands-on project to go from zero to a working self-driving robot, built by a non-engineer with AI as the entire support team. The goal is genuine fluency in how autonomous systems *perceive, decide, and act* — the same mental model behind real AV companies, at a $150 scale.

## Demo

The rover driving a full lap **on its own** — camera → trained neural network → steering, running live on a Raspberry Pi:

https://github.com/user-attachments/assets/d04fd80b-f6ff-4a6d-bd47-b589e957710e



## The core idea: sense → think → act

Every autonomous machine, from this rover to a robotaxi, runs the same loop dozens of times per second:

- **Sense** — a camera and distance sensor turn the world into data
- **Think** — code (later, a trained neural network) turns data into a driving decision
- **Act** — motors and a steering servo turn the decision into motion

Master that loop on a small rover and the concepts scale directly to full-size autonomy.

## Hardware

- **Brain:** Raspberry Pi 3 Model B (single-board computer)
- **Platform:** SunFounder PiCar-X kit — chassis, drive motor, steering servo, camera, ultrasonic + line sensors
- **Training compute:** Google Colab (cloud) — no local GPU required

## Roadmap

| Phase | Goal | Key concepts | Status |
|-------|------|--------------|--------|
| 0 · Setup | Tools, repo, order parts | control loop, SBC, version control | ✅ |
| 1 · Make it move | Assemble; drive by code | motors, PWM, servos, actuators, headless control | ✅ |
| 2 · Make it see | Camera + rule-based autonomy | computer vision, OpenCV, line following, obstacle avoidance | ✅ |
| 3 · Make it learn | Train a neural net to drive itself | data collection, training, inference, end-to-end learning | 🟢 in progress |
| 4 · Extend & document | One upgrade (3D printing / better sensor / ROS 2) + write-up | (varies) | ⬜ |

## Status

🟢 **Phase 3 — it drives itself.** The rover collects driving data by manual teleop, trains a **PilotNet** CNN (NVIDIA DAVE-2 architecture) in the cloud, and runs the model on-device (TF-Lite) to steer itself around a track in real time. Recent work: diagnosed a mid-corner failure as regression-to-the-mean on thin data, and fixed it with **DAgger** (targeted correction data) plus **lighting augmentation** for robustness to changing light. Follow along in the commit history and project notes.

---

*Built as a learning project, with AI as PM, instructor, and engineering support.*
