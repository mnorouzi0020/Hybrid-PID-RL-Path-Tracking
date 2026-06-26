# Hybrid PID–RL Path Tracking Controller for Autonomous Vehicles

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](YOUR_ARXIV_LINK_HERE)

> **A novel hybrid control architecture that combines PPO-based Reinforcement Learning with a PID controller for robust autonomous vehicle path tracking under low-friction road conditions.**

---

## Abstract

Path tracking controllers are crucially important in the performance and safety of autonomous vehicles. These controllers’ main duty is enabling the vehicle to accurately follow a predefined trajectory while maintaining stability and avoiding deviations, which becomes even more challenging in varying road and weather conditions, where maintaining accuracy and safety is critical. The controller must be able to handle sharp turns and adapt to varying road conditions, such as icy or rainy roads, which plays a significant role in assessing the efficacy of a path-tracking controller. The PID controller is widely used for its acceptable performance with fixed gains, while reinforcement learning-based controllers, such as the extensively used PPO algorithm, offer adaptive and accurate results through environmental interaction. However, when considering RL controllers, we face two major challenges: parameter tuning and fluctuating control signals. To address these issues, we propose a novel combination. One modifies the neural network structure, and the other incorporates two alterations in the algorithm’s loss function, resulting in parameter robustness and reduction in control signal fluctuations. Then, we evaluate and compare the performance of the PID controller and the RL-based controller. Our comparison focuses on path-tracking accuracy, highlighting the trade-offs between traditional and adaptive control methods under challenging conditions. Initial results show that the RL-based controller performs worse than the PID with fixed gains and reference speed, while the fixed PID lacks high path-tracking accuracy. Moreover, we propose a hybrid approach where RL optimizes the PID controller for adaptive gain and reference speed adjustment. Results show that while the fixed PID fails in low-friction conditions, the PID-RL combination outperforms each method alone, achieving superior performance in both normal and challenging road conditions. These results illustrate the potential of hybrid approaches in optimizing control systems for autonomous vehicles, particularly in enhancing path-tracking performance under varying road conditions.

---

## Key Contributions

| Contribution | Description |
|---|---|
| ** Combination of LipsNet and CAPS** | Lipschitz-constrained actor network that enforces bounded output sensitivity, improving parameter robustness, is combined with Temporal (LT) and spatial (LS) smoothness penalties added to the PPO loss function to reduce control signal chattering|
| **Hybrid Architecture** | RL adaptively tunes PID gains (Kp, Ki, Kd) and reference speed in real time, combining the interpretability of PID with the adaptability of RL |
| **Low-Friction Evaluation** | Systematic comparison under reduced tire–road friction (Fry scaled), where the fixed PID fails and the hybrid succeeds |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                          State                           │
│  [position, heading, Vx, Vy, β, αf, αr, ω,               │
│   pos_error, ang_error, checkpoints, future path]        │
└────────────────────────────┬─────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   LipsNet Actor │  ← Lipschitz-constrained
                    │   (PPO / GAE)   │    via KNet + Jacobian norm
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │   Action (4-dim)            │
              │  [Ref. Speed, Kp, Ki, Kd]   │
              └──────────────┬──────────────┘
                             │
            ┌────────────────▼────────────────┐
            │       PID Steering Controller   │
            │   δ = Kp·e_pos + Ki·∫e + Kd·ė   │
            └────────────────┬────────────────┘
                             │
            ┌────────────────▼────────────────┐
            │   Non-Linear Bicycle Model      │
            └─────────────────────────────────┘
```

### PPO Loss Function (with CAPS)

```
L_total = L_critic − L_actor − β·H(π) + λT·LT + λS·LS + λK·‖K‖²
```

Where `LT` is the temporal smoothness penalty (Jeffrey's divergence between consecutive actions)
and `LS` is the spatial smoothness penalty (divergence between actions at nearby states).

---

## Vehicle Model

The vehicle is simulated using a **Non-Linear Bicycle Model**:

| Parameter | Value |
|---|---|
| Wheelbase (L) | 2.9 m |
| Mass (m) | 1500 kg |
| Yaw inertia (Iz) | 2250 kg·m² |
| Front cornering stiffness (Cf) | 3200 N/rad |
| Rear cornering stiffness (Cr) | 3400 N/rad (scaled by 0.05 for low-friction) |
| Time step (dt) | 0.1 s |
| Max steering angle | ±40° |

State variables tracked: `Vx, Vy, β (sideslip), αf (front slip), αr (rear slip), ω (yaw rate)`

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/mnorouzi0020/Hybrid-PID-RL-Path-Tracking.git
cd mnorouzi0020

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Training

```python
# In main.py, set:
mode = True          # Training mode (Training)
save_model = True    # Save weights after training
load_model = False   # Start fresh

python main.py
```

Training logs vehicle position, speed, steering, Kp/Ki/Kd gains, and sideslip angle (β)
per episode. TensorBoard logging is supported.

### Inference

```python
# In main.py, set:
mode = False         # Inference mode
save_model = False
load_model = True    # Load pre-trained weights

python main.py
```

Pre-trained weights should be placed in `saved_models/`.

### Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `learning_rate` | 1e-4 | Adam optimizer LR |
| `gamma` | 0.90 | Discount factor |
| `lambda_` | 0.95 | GAE lambda |
| `batch_size` | 64 | Mini-batch size |
| `update_freq` | 1024 | Steps between PPO updates |
| `num_epochs` | 30 | Gradient update epochs per batch |
| `policy_kl_range` | 0.03 | KL threshold for PPO clipping |
| `lam_T` | 1e-4 | Temporal CAPS weight |
| `lam_S` | 5e-4 | Spatial CAPS weight |
| `lamda_k` | 1e-5 | LipsNet K-regularisation weight |
| `Kp_max / Ki_max / Kd_max` | 100 / 1 / 100 | PID gain output ranges |

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── LICENSE
├── main.py                    ← Entry point (training & inference)
├── saved_models/              ← Pre-trained weights (actor, critic)
│   └── .gitkeep
└── results/                   ← Plots and evaluation outputs
    └── .gitkeep
```

---

## Results Summary

| Controller | Normal Road | Low-Friction Road |
|---|---|---|
| Fixed PID | ✅ Acceptable | ❌ Fails |
| PPO (RL only) | ⚠️ Worse than PID (fixed gains) | ⚠️ Unstable |
| **PID–RL (Hybrid)** | **✅ Best** | **✅ Outperforms both** |

The hybrid approach achieves the lowest cross-track error and smoothest control signals across
all tested road conditions, particularly excelling when friction coefficients are low.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{norouzi2026reinforcement,
  title={Reinforcement learning-based controller for path tracking of an autonomous vehicle},
  author={Norouzi, Mojtaba and Poshtan, Javad},
  journal={Proceedings of the Institution of Mechanical Engineers, Part D: Journal of Automobile Engineering},
  volume={240},
  number={6},
  pages={3340--3363},
  year={2026},
  publisher={SAGE Publications Sage UK: London, England}
}
```

---

## Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Core libraries:

- [TensorFlow 2.x](https://www.tensorflow.org/) — Neural network training
- [TensorFlow Probability](https://www.tensorflow.org/probability) — Stochastic policy distributions
- [NumPy](https://numpy.org/) — Numerical computation
- [Matplotlib](https://matplotlib.org/) — Visualisation
- [SciPy](https://scipy.org/) — Optimisation utilities
- [scikit-learn](https://scikit-learn.org/) — Preprocessing

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

Mojtaba Norouzi

📧 mojtaba_norouzi77@alumni.iust.ac.ir

🔗 [LinkedIn](www.linkedin.com/in/mojtaba-norouzi-2522a0206)


---
