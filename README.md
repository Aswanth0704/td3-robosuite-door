# TD3 on RoboSuite Door (Panda)

This repository implements **Twin Delayed Deep Deterministic Policy Gradient (TD3)** for robotic manipulation on the **RoboSuite Door** environment using the **Panda** robot.

The code is designed for **HPC training (SLURM)** and follows best practices for:
- reproducible reinforcement learning
- large-scale robotic simulation
- clean Git repositories

---

## Overview

- **Algorithm**: TD3 (Actor–Critic, twin critics, delayed policy updates)
- **Environment**: RoboSuite `Door`
- **Robot**: Panda
- **Controller**: RoboSuite Composite Controller (`BASIC`)
- **Framework**: PyTorch + GymWrapper
- **Training target**: Continuous control, sparse-to-shaped reward manipulation task

---

## Features

- ✅ TD3 with twin critics
- ✅ Target policy smoothing
- ✅ Delayed actor updates
- ✅ Polyak averaging (soft target updates)
- ✅ Replay buffer
- ✅ TensorBoard logging
- ✅ RoboSuite + Gymnasium compatibility

---

## Repository Structure

```text
.
├── main.py              # Training entry point
├── td3_torch.py         # TD3 agent implementation
├── networks.py          # Actor and Critic neural networks
├── buffer.py            # Replay buffer
├── test.py              # Environment sanity test
├── requirements.txt     # Python dependencies
├── README.md
└── .gitignore
