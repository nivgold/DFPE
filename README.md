# DFPE: A Diverse Fingerprint Ensemble for Enhancing LLM Performance - Official Implementation

![proposed framework](https://i.ibb.co/j9J4k3Hf/DFPE-figure.jpg)

## Abstract
> Large Language Models (LLMs) have shown remarkable capabilities across various natural language processing tasks but often struggle to excel uniformly in diverse or complex domains. We propose a novel ensemble method - Diverse Fingerprint Ensemble (DFPE), which leverages the complementary strengths of multiple LLMs to achieve more robust performance. Our approach involves: (1) clustering models based on response "fingerprints" patterns, (2) applying  a quantile-based filtering mechanism to remove underperforming models at a per-subject level, and (3) assigning adaptive weights to remaining models based on their subject-wise validation accuracy. In experiments on the Massive Multitask Language Understanding (MMLU) benchmark, DFPE outperforms the best single model by 3% overall accuracy and 5% in discipline-level accuracy. This method increases the robustness and generalization of LLMs and underscores how model selection, diversity preservation, and performance-driven weighting can effectively address challenging, multi-faceted language understanding tasks.


## Getting Started

### Prerequisites
- Python >= 3.8
- PyTorch >= 2.0
- CUDA-compatible GPU (optional but recommended)


### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/.../dfpe.git
   cd dfpe
   ```
2. Install the dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running DFPE

The main script for training and evaluation is `main.py`. Below is a description of the required and optional arguments:

```bash
python main.py --help
```

**Arguments:**

**Required:**
- `--models`, `-m`  
  LLM models list separated by commas (e.g., "model1,model2,model3").
- `--question-column`, `-q`  
  Name of the question column in the dataset.
- `--choices-column`, `-c`  
  Name of the choices column in the dataset.
- `--subject-column`, `-cat`  
  Name of the subject column in the dataset.
- `--label-column`, `-l`  
  Name of the label column in the dataset.
- `--embedding-model`, `-e`  
  Embedding model to use for encoding the dataset.

**Optional:**
- `--hf-dataset-name`, `-hd`  
  Hugging Face dataset name (if applicable).
- `--dbscan-epsilon`, `-eps`  
  DBSCAN epsilon value for clustering. Default: 0.5.
- `--quantile-threshold`, `-qt`  
  Quantile threshold for filtering data. Default: 0.75.
- `--scaling-factor`, `-s`  
  Scaling factor for normalization or transformation. Default: 1.0.
- `--batch-size`, `-b`  
  Batch size for model inference. Default: 32.

### Example Command

```bash
python main.py \
  --models "Qwen/Qwen2.5-3B-Instruct,microsoft/Phi-3.5-mini-instruct" \
  --hf-dataset-name "cais/mmlu" \
  --question-column "question" \
  --choices-column "choices" \
  --subject-column "subject" \
  --label-column "answer" \
  --embedding-model "all-MiniLM-L6-v2" \
  --dbscan-epsilon 0.7 \
  --quantile-threshold 0.8 \
  --scaling-factor 1.2 \
  --batch-size 16
```

## Repository Stracture
```text
.
├── src/                       # Source Code Directory
    ├── arguments_manager.py   # Arguments Management
    ├── dataset.py             # Dataset Wrapper
    ├── ensemble.py            # Ensembling Predictions
    ├── inference.py           # Text Generatoin Loop
    ├── main.py                # Main
    ├── model.py               # Model Wrapper
    └── utils.py               # General-Purpose Utils
├── requirements.txt           # Required Python packages
└── README.md                  # Project README
```
