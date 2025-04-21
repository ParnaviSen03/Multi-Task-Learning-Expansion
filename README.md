# Multi-Task Sentence Transformer

This project implements a multi-task learning model using a shared BERT encoder to perform:

- **Task A:** Sentiment Classification (SST-2 dataset)
- **Task B:** Question Detection (synthetic heuristic labels)

Developed as part of a take-home assessment to demonstrate transformer-based modeling, multi-task learning design, training strategies, and performance evaluation.

---

## Tasks Breakdown

###  Task 1: Sentence Transformer
- Used `bert-base-uncased` from Hugging Face Transformers.
- Generated sentence embeddings via **mean pooling** over token embeddings.
- Tested on longer SST-2 sentences to validate output shapes and quality.

###  Task 2: Multi-Task Learning Expansion
- Added two **task-specific linear heads** for sentiment classification and question detection.
- Used a shared BERT encoder and dynamic routing based on a `task` flag in the `forward()` method.
- Task B labels were synthetically generated using a heuristic function (e.g., if sentence starts with “what”, “how”, or ends with `?`).

###  Task 3: Training Considerations
- **Freezing the entire model:** Fast and memory-efficient but limits adaptation to the task.
- **Freezing just the BERT encoder:** Enables the heads to adapt, good trade-off for small data.
- **Freezing one task head:** Useful in multi-task learning to avoid negative transfer once one task has converged.

**Transfer learning strategy:**
- Freeze BERT layers 0–6 and fine-tune layers 7–11 + task heads.
- Leverages general language features while adapting higher layers to new tasks.

###  Task 4: Training Loop & Visualization
- Implemented a training loop using PyTorch with joint optimization for both tasks.
- Used `CrossEntropyLoss` and `Adam` optimizer.
- Tracked loss and accuracy for each task across epochs.
- Visualized learning progression using Matplotlib.

---

##  Accuracy: Before vs. After Training

| Phase              | Task A Accuracy | Task B Accuracy |
|-------------------|-----------------|-----------------|
| **Before Training** | 0.5230          | 0.5630          |

| Epoch | Task A Accuracy | Task B Accuracy |
|-------|-----------------|-----------------|
| 1     | 0.7560          | 0.9290          |
| 2     | 0.9260          | 0.9570          |
| 3     | **0.9730**      | **0.9890**      |

 The model showed significant improvements after training, especially in Task A, which started near random.

---

##  Visualizations

Training curves were generated to show task-specific loss and accuracy:

- **Loss vs Epoch:** Both tasks show consistent reduction in loss over time.
- **Accuracy vs Epoch:** Accuracy steadily increases and saturates, indicating learning convergence.

Plots are included at the end of the notebook.

---

##  How to Run

###  Option 1: Run the Notebook Locally

1. Install all required dependencies:

```bash
pip install -r requirements.txt

2. Launch the notebook:

jupyter notebook multitask_training.ipynb
