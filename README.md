# 🤖 Self-Driving Rover

Learning robotics, autonomous vehicles, and physical AI by building a small camera-equipped rover from parts — and teaching it to drive itself.

This repo documents a hands-on project to go from zero to a working self-driving robot, built by a non-engineer with AI as the entire support team. The goal is genuine fluency in how autonomous systems *perceive, decide, and act* — the same mental model behind real AV companies, at a $150 scale.

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

| Phase | Goal | Key concepts |
|-------|------|--------------|
| 0 · Setup | Tools, repo, order parts | control loop, SBC, version control |
| 1 · Make it move | Assemble; drive by code | motors, PWM, servos, actuators, headless control |
| 2 · Make it see | Camera + rule-based autonomy | computer vision, OpenCV, line following, obstacle avoidance |
| 3 · Make it learn | Train a neural net to drive itself | data collection, training, inference, end-to-end learning |
| 4 · Extend & document | One upgrade (3D printing / better sensor / ROS 2) + write-up | (varies) |

## Status

🟢 **Phase 1 — assembling the rover.** Follow along in the commit history and project notes.

---

*Built as a learning project, with AI as PM, instructor, and engineering support.*
