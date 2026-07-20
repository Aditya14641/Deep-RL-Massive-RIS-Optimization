# RL-Adaptive-PID-Control
### *Reinforcement Learning Framework for Adaptive Tuning of Unstable Systems*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Stable--Baselines3-orange.svg)
![Build](https://img.shields.io/badge/Build-Reproducible-green.svg)

## 📌 Project Overview
This repository contains a comprehensive comparative study of Reinforcement Learning (RL) algorithms applied to the problem of **Adaptive PID Control**. The target system is an inherently unstable first-order plant:

$$G(s) = \frac{1}{s-1}$$

The project demonstrates how modern Deep RL agents can learn to dynamically adjust PID gains ($K_p, K_i, K_d$) to stabilize the plant, even when high aggressive control is required to overcome inherent instability.

---

## 🧠 Algorithm Overview
The project benchmarks several Reinforcement Learning architectures in a continuous control task.

| Algorithm | Type | Description |
| :--- | :--- | :--- |
| **PPO** | On-Policy | Proximal Policy Optimization. Uses clipped updates for stable training. |
| **SAC** | Off-Policy | Soft Actor-Critic. Uses entropy regularization for maximum exploration. |
| **DDPG** | Off-Policy | Deep Deterministic Policy Gradient. Efficient for high-dimensional action spaces. |
| **A2C** | On-Policy | Advantage Actor-Critic. A synchronous, baseline policy gradient method. |

---

## 🛠️ Theoretical Framework

### 1. The Control Problem
In this framework, the RL agent acts as the "Higher-Level Controller." 
* **State ($S$):** $\{e, \int e, \dot{e}\}$
* **Action ($A$):** $\{K_p, K_i, K_d\}$
* **Plant:** $\dot{y} = y + u$ (Time-domain equivalent of $1/(s-1)$)

### 2. Performance Metrics
To evaluate the controller, we calculate standard control engineering metrics:
* **ISE:** Integral Squared Error
* **IAE:** Integral Absolute Error
* **SSE:** Steady-State Error

---

## 📊 Experimental Results
All models were trained for **100,000 timesteps** with a sampling time of $dt = 0.1s$ and a target setpoint of $1.0$.

### Final Performance Comparison
| Algorithm | Rise Time (s) | Settling Time (s) | Overshoot (%) | ISE | IAE | SSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PPO** | 1.100 | 3.700 | 85.67% | 1.199 | 1.385 | 0.0000 |
| **SAC** | 1.100 | 4.900 | **42.40%** | **1.155** | 1.480 | 0.0000 |
| **DDPG** | 1.100 | **2.400** | 52.49% | 1.181 | **1.341** | 0.0000 |
| **A2C** | 1.100 | 4.200 | 99.55% | 1.225 | 1.425 | 0.0000 |

### Key Observations
* **Rise Time Consistency:** All algorithms achieved a consistent Rise Time of **1.1s**, suggesting they reached the maximum allowable control effort early to stabilize the unstable pole.
* **DDPG Efficiency:** **DDPG** provided the best balance of speed, achieving the fastest **Settling Time (2.4s)** and the lowest **IAE (1.341)**.
* **SAC Stability:** While slower to settle, **SAC** demonstrated the best transient control with the lowest **Overshoot (42.40%)** and the best **ISE (1.155)**, highlighting the benefits of its entropy-based exploration.
* **Precision:** All algorithms achieved an **SSE of 0.0000**, confirming successful zero-error tracking.

---

## 🚀 Getting Started
1. **Clone the Repo:** `git clone https://github.com/Aditya14641/RL-Adaptive-PID-Control.git`
2. **Install Deps:** `pip install -r requirements.txt`
3. **Run Pipeline:** `bash scripts/run.sh`

---

## 🎓 Academic Credit
**Aditya Mehta | Krishna Gopal Rathi**
*Undergraduate Students, The LNM Institute of Information Technology (LNMIIT)*
