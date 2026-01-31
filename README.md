# FedSCOPE: Federated Cross-Domain Sequential Recommendation

## 📖 Introduction

FedSCOPE is a federated learning framework designed for Cross-Domain Sequential Recommendation (CDSR). It addresses three key challenges: data sparsity, domain heterogeneity, and the trade-off between privacy and utility.

Key components implemented in this project:

1. **Offline LLM-based Semantic Augmentation**: Enhances item/user representations using semantic vectors (simulated via Sentence-BERT) to address sparsity without exposing raw data.
2. **IIDCL (Intra- and Inter-Domain Decoupled Contrastive Learning)**: Aligns user representations across domains while maintaining intra-domain personalization.
3. **Adaptive Personalized Differential Privacy (DP)**: dynamically allocates privacy budgets and clips gradients based on client data scale to balance protection and model performance.

## 📂 Project Structure

The code is modularized into the following files:

```text
FedSCOPE/
├── config.py       # Configuration of hyperparameters (Model, Training, Privacy)
├── dataset.py      # Data loading, Preprocessing, and Offline Semantic Augmentation
├── model.py        # Backbone model (SASRec Encoder) and Feature Fusion modules
├── losses.py       # Implementation of IIDCL (Contrastive Losses)
├── federated.py    # Server/Client logic, Local Training, and Adaptive DP
├── main.py         # Main entry point to run the federated simulation
└── README.md       # This instruction file

```

## ⚙️ Requirements

To run this code, you need Python 3.8+ and the following libraries. You can install them via pip:

```bash
pip install torch numpy pandas sentence-transformers

```

* **PyTorch**: Core deep learning framework.
* **Sentence-Transformers**: Used to generate semantic embeddings for items (simulating the LLM augmentation process described in the paper).
* **Pandas/Numpy**: Data manipulation.

## 🚀 How to Run

### 1. Data Preparation (Optional but Recommended)

The code is designed to work with the **Amazon Review Dataset** (specifically *Movies_and_TV* and *Books*).

1. Download the **5-core** datasets from the [UCSD Amazon Data Repository](https://nijianmo.github.io/amazon/index.html).
* `reviews_Movies_and_TV_5.json.gz`
* `reviews_Books_5.json.gz`


2. Place these files in the project root directory.
3. Update the file paths in `config.py` if necessary.

> **Note:** If these files are not found, the code will automatically generate **Mock Data** (synthetic user/item interactions) so you can run the code immediately to verify the pipeline.

### 2. Execution

Run the main script to start the federated training simulation:

```bash
python main.py

```

### 3. Output

The script will simulate the Federated Learning process:

* Data partitioning among clients.
* Offline semantic feature generation.
* Round-by-round training logs (Loss values).
* Global aggregation updates.

## 🔍 Code to Paper Mapping

This implementation strictly follows the methodology described in the paper. Here is the mapping between the code modules and the paper's equations:

| Paper Component | Description | Paper Section / Eq | Code Location |
| --- | --- | --- | --- |
| **LLM Augmentation** | Generating semantic attributes and embeddings | Sec 3.2 / Eq (2) | `dataset.py` -> `generate_semantic_embeddings` |
| **Feature Fusion** | Fusing ID-based and Semantic embeddings | Sec 3.3 / Eq (3) | `model.py` -> `FeatureFusion` |
| **Backbone** | Sequential Encoder (SASRec) | Sec 3.3 | `model.py` -> `SASRecEncoder` |
| **IIDCL (Intra)** | Contrastive learning within the same domain | Sec 3.4 / Eq (4) | `losses.py` -> `iidcl_loss` |
| **IIDCL (Inter)** | Contrastive learning across domains | Sec 3.4 / Eq (6) | `losses.py` -> `iidcl_loss` |
| **Total Loss** | Prediction + Contrastive Loss | Eq (8) | `federated.py` -> `Client.train_round` |
| **Adaptive Budget** | Dynamic epsilon based on data size | Eq (13) | `federated.py` -> `apply_adaptive_dp` |
| **Adaptive Clipping** | Dynamic gradient clipping | Eq (11) | `federated.py` -> `apply_adaptive_dp` |
| **Noise Injection** | Adding Gaussian noise | Eq (10) | `federated.py` -> `apply_adaptive_dp` |

## 🛠 Configuration (`config.py`)

You can modify `config.py` to experiment with different settings:

* **`num_clients`**: Number of federated clients (e.g., 5 for demo, 50+ for real experiments).
* **`dp_epsilon_total`**: The total privacy budget.
* **`alpha`, `lambda_intra`, `lambda_inter**`: Weights for the loss functions.
* **`domain_a_path` / `domain_b_path**`: Paths to your specific datasets.

## ⚠️ Implementation Notes

* **LLM Simulation**: In the paper, a Large Language Model (like GPT) is used to infer attributes (Director, Genre, etc.) which are then embedded. In this implementation, to make it runnable locally without API costs, we use `sentence-transformers` (MiniLM) to directly encode the raw item metadata/text. This is mathematically equivalent (both produce a semantic vector ).
* **Device**: The code automatically detects CUDA (GPU) or CPU.
