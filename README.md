# DeepRM_2 — CNN-Based Resource Management with Anomaly Detection in the Cloud-Edge Continuum

This repository builds a **CNN-based policy network** on top of the [DeepRM_ECO](https://dagshub.com/n.barring/DeepRM_ECO.git) framework — itself an extension of the original [DeepRM](http://people.csail.mit.edu/hongzi/content/publications/DeepRM-HotNets16.pdf) (HotNets'16) deep reinforcement learning scheduler.

The original DeepRM uses policy gradient methods to learn scheduling policies for a single machine with multiple resources. A prior extension, **DeepRM_ECO**, extended DeepRM with:

- **Three heterogeneous compute nodes** (2 edge nodes + 1 cloud node) instead of a single machine
- **Anomalous job modeling** — jobs with abnormally long durations or high resource demands
- **Anomaly-aware scheduling policies** that route jobs to appropriate nodes based on job characteristics
- **Multiple baseline heuristics** for comparison: SJF, FCFS, Weighted Round Robin (WRR), Packer, and Random
- **Anomaly detection module** — proactively identifies and treats anomalous workloads to improve fault-tolerance

**DeepRM_2** extends DeepRM_ECO by introducing a **Convolutional Neural Network (CNN)** architecture for the policy network, replacing the original fully connected (dense) network. The CNN is better suited to exploit the spatial structure of the image-based state representation, where resource utilization across machines is encoded as a 2D grid. Inspired by [DeepRM_Plus](https://doi.org/10.1109/JIOT.2020.3025015) (Guo et al., 2020), which demonstrated the benefits of CNN and imitation learning for resource scheduling, DeepRM_2 synthesises these modelling improvements with the anomaly detection capabilities of DeepRM_ECO.

---

## 🏗️ Architecture Overview

The system models a resource management environment where incoming jobs must be scheduled across three machines with different resource capacities:

| Node | Type | Resource Slots | Role |
|------|------|---------------|------|
| Machine 1 | Edge | 10 slots | Handles normal short jobs |
| Machine 2 | Edge | 10 slots | Handles normal short jobs |
| Machine 3 | Cloud | 15 slots | Handles long-duration and resource-intensive (anomalous) jobs |

The scheduling agent observes the current system state — represented as an image of resource utilization across all three machines, the pending job queue, and the backlog — and selects one of `num_nw × 3 + 1` actions (allocate a job to one of the three machines, or hold).

---

## 🔄 Training Workflow

The training pipeline follows a two-phase approach:

1. **Supervised Learning (Imitation):** The agent learns to mimic a heuristic policy (SJF, FCFS, or WRR) via supervised training on state-action pairs. This provides a warm-start for the policy network.

2. **Policy Gradient Reinforcement Learning:** Starting from the supervised pre-trained weights, the agent is fine-tuned using REINFORCE with RMSProp optimization. The reward signal penalizes delays, holding jobs, and backlog overflow, with additional penalties for misrouting anomalous jobs.

### Network Architecture

Two network variants are available, selectable via the `--use_cnn` flag:

- **Dense Network (`--use_cnn=False`):** The original DeepRM_ECO architecture — a simple fully connected network with ReLU activation (20 hidden units) and softmax output.
- **CNN Network (`--use_cnn=True`):** The main contribution of this thesis (**DeepRM_2**) — processes the state space as an image input through a 5-layer CNN:

| Layer | Filter | Filters | Activation | Output |
|-------|--------|---------|------------|--------|
| Input | — | — | — | (30, 225, 1) |
| Conv2D | (3, 3) | 16 | ReLU | (28, 223, 16) |
| MaxPool | (2, 2) | — | — | (14, 111, 16) |
| Dense | — | — | ReLU | 64 |
| Dropout | — | — | — (p=0.5) | 64 |
| Output | — | — | Softmax | 16 |

Both networks are built with [Lasagne](https://lasagne.readthedocs.io/) on top of [Theano](http://deeplearning.net/software/theano/).

### Reward Function

The reward function simultaneously incentivises minimising job slowdown and correctly detecting anomalies:

$$R = \sum_{j \in J} \frac{-1}{T_j} + \sum_{j \in C} \begin{cases} +3 & \text{if } T_j \geq A_{len} \\ -2 & \text{otherwise} \end{cases}$$

Where $J$ represents all jobs in the backlog, waiting queue, and edge instances (M1, M2), $C$ represents jobs on the cloud instance (M3), $T_j$ is the duration of job $j$, and $A_{len}$ is the anomaly length threshold. The agent is rewarded for correctly migrating anomalous jobs to the cloud and penalised for sending normal jobs there.

---

## ✨ Key Features

- **CNN Policy Network:** Introduces a convolutional architecture that exploits the 2D spatial structure of the state representation, compared to the flat dense network used in DeepRM_ECO.
- **Multi-Machine Scheduling:** Extends DeepRM from a single machine to three heterogeneous compute nodes (cloud + edge), expanding the action space to support allocation decisions across machines.
- **Anomalous Job Modeling:** A bimodal job distribution with configurable anomaly rates, generating jobs with abnormal durations (19–29 time steps) and/or high resource demands (11–15 slots).
- **Anomaly-Aware Heuristics:** Custom implementations of SJF, FCFS, and WRR that route jobs to appropriate nodes based on job length and resource thresholds.
- **Comprehensive Evaluation:** Automated testing across varying workloads (simulation lengths 10–200), anomaly rates (0.10, 0.25, 0.50), and job slowdown analysis with CDF plots.
- **Data Collection & Metrics:** Built-in CSV logging of allocated jobs per machine, test metrics (average/total slowdown), and parameter snapshots in YAML format.
- **SLURM Integration:** Ready-to-use job scripts for training and testing on HPC clusters with GPU support.

---

## 🚀 Getting Started

### 🛠️ Prerequisites

- **Python 2.7** (required by Theano/Lasagne)
- **Conda** (recommended for environment management)
- **GPU** (optional, for faster training — CPU mode is the default)

### ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd DeepRM_2
   ```

2. **Create the Conda environment:**
   ```bash
   conda env create -f environment.yaml
   conda activate py27
   ```

   Or install manually:
   ```bash
   sudo apt-get update
   sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
   pip install --user Theano==0.8.0
   pip install --user Lasagne==0.1
   pip install -r requirements.txt
   ```

3. **Create the data directory:**
   ```bash
   mkdir -p data
   ```

---

## 📈 Usage

All experiments are launched through `launcher.py`. The default parameters are defined in `parameters.py`.

### Supervised Learning (Imitation Pre-training)

Train the policy network to imitate a heuristic baseline (default: WRR):

```bash
python launcher.py --exp_type=pg_su --simu_len=50 --num_ex=100 --ofile=data/pg_su --out_freq=10 --num_epochs=101
```

### Policy Gradient Reinforcement Learning

Fine-tune the pre-trained network using REINFORCE:

```bash
# CNN variant (DeepRM_2)
python launcher.py --exp_type=pg_re --pg_re=data/pg_su_net_file_100.pkl --simu_len=10 --num_ex=50 --ofile=data/pg_re --num_epochs=1001 --use_cnn=True

# Dense variant (DeepRM_ECO)
python launcher.py --exp_type=pg_re --pg_re=data/pg_su_net_file_100.pkl --simu_len=10 --num_ex=5 --ofile=data/pg_re --num_epochs=101 --use_cnn=False
```

### Testing & Evaluation

Compare the trained agent against heuristic baselines on unseen workloads:

```bash
# Test CNN model
python launcher.py --exp_type=test --simu_len=50 --num_ex=20 --pg_re=experiments/DeepRM_2/simu10,numex5,epoch1000/pg_re_1000.pkl --unseen=True --use_cnn=True

# Test Dense/ECO model
python launcher.py --exp_type=test --simu_len=50 --num_ex=20 --pg_re=experiments/DeepRM_ECO/simu10,numex5,epoch1000/pg_re_1000.pkl --unseen=True --use_cnn=False
```


### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--exp_type` | Experiment type: `pg_su`, `pg_re`, `test` | `pg_re` |
| `--num_res` | Number of resource dimensions | `2` |
| `--num_nw` | Number of visible new work slots | `5` |
| `--simu_len` | Simulation length (busy cycle) | `50` |
| `--num_ex` | Number of example sequences | `15` |
| `--num_epochs` | Number of training epochs | `3001` |
| `--time_horizon` | Look-ahead time horizon (screen height) | `30` |
| `--res_slot` | Total resource slots (screen width) | `15` |
| `--max_job_len` | Maximum new job duration | `18` |
| `--max_job_size` | Maximum job resource request | `10` |
| `--new_job_rate` | New job arrival rate (Poisson λ) | `0.7` |
| `--lr_rate` | Learning rate | `0.001` |
| `--dist` | Discount factor | `0.95` |
| `--pg_re` | Path to pre-trained policy network `.pkl` file | `None` |
| `--ofile` | Output file prefix | `data/tmp` |
| `--out_freq` | Checkpoint/output frequency (epochs) | `10` |
| `--unseen` | Use unseen random seed for testing | `False` |
| `--use_cnn` | Use CNN architecture (`True`) or Dense (`False`) | `True` |
| `--render` | Plot environment dynamics | `False` |

---

## 📊 Analyzing Results

The `results/` and `archive/results_slowdown/` directories contain Jupyter notebooks for analyzing experimental outcomes:

- **Slowdown Analysis:** CDF plots comparing job slowdown across DeepRM variants and heuristic baselines
- **Anomaly Analysis:** Impact of anomaly rates (10%, 25%, 50%) on scheduling performance
- **Workload Analysis:** Performance across varying simulation lengths and job arrival rates

Test metrics are saved as CSV files with columns: `Test Type`, `Average Slowdown`, `Total Slowdown`, `Workload`, `Dist Proba`, `Anomaly Rate`.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 📄 Master Thesis Paper

This repository is the official implementation for the paper "**Learning-Based Anomaly-Tolerant Resource Management in Cloud-Edge Continuum**". For a detailed explanation of the methodology, experimental setup, and results, please refer to the paper.

➡️ [Read the full paper (PDF)](paper.pdf)